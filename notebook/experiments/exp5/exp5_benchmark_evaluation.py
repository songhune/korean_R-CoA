"""
KLSBench Evaluation Framework
한국 고전 문헌 벤치마크 평가 프레임워크

지원 모델:
1. 오픈소스 모델: Llama, Qwen, EXAONE 등
2. 비공개 API 모델: GPT-4, Claude, Gemini 등
3. 지도학습 모델: GwenBert, Tongu 등

평가 태스크:
- Classification: 문체 분류
- Retrieval: 출처 식별
- Punctuation: 구두점 복원
- NLI: 자연언어추론
- Translation: 번역
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import re
from tqdm import tqdm
import numpy as np
import unicodedata

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Configuration loader
try:
    from config_loader import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] config_loader not available, using default paths")
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class KLSBenchEvaluator:
    """KLSBench 벤치마크 평가기"""

    def __init__(self,
                 benchmark_path: str,
                 output_dir: str,
                 model_type: str = "api",  # "api", "opensource", "supervised"
                 max_samples_per_task: Optional[int] = None,
                 sample_ratio: Optional[float] = None,
                 temperature: float = 0.0,
                 save_samples: bool = True,
                 num_samples_to_save: int = 5):
        """
        Args:
            benchmark_path: 벤치마크 JSON 파일 경로
            output_dir: 결과 저장 디렉토리
            model_type: 모델 타입 (api/opensource/supervised)
            max_samples_per_task: 태스크당 최대 샘플 수 (None이면 전체)
            sample_ratio: 샘플링 비율 (0.0~1.0, None이면 전체)
            temperature: 생성 temperature (0.0~1.0)
            save_samples: 샘플 출력 저장 여부 (appendix용)
            num_samples_to_save: 저장할 샘플 개수
        """
        self.benchmark_path = Path(benchmark_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.max_samples_per_task = max_samples_per_task
        self.sample_ratio = sample_ratio
        self.temperature = temperature
        self.save_samples = save_samples
        self.num_samples_to_save = num_samples_to_save

        # 벤치마크 로드
        self.load_benchmark()

        # 프롬프트 템플릿
        self.setup_prompts()

    def load_benchmark(self):
        """벤치마크 데이터 로드"""
        print(f"[LOAD] Benchmark: {self.benchmark_path}")
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)

        print(f"  Benchmark: {self.benchmark['benchmark_info']['name']}")
        print(f"  Total items: {self.benchmark['benchmark_info']['total_size']:,}")
        print(f"  Tasks: {len(self.benchmark['tasks'])}")

        # Apply sampling limits (priority: max_samples > sample_ratio)
        if self.max_samples_per_task:
            print(f"\n[SAMPLING] Limited to {self.max_samples_per_task} samples per task")
            for task_name, task_data in self.benchmark['tasks'].items():
                original_size = len(task_data['data'])
                task_data['data'] = task_data['data'][:self.max_samples_per_task]
                task_data['size'] = len(task_data['data'])
                print(f"  - {task_name}: {original_size} -> {task_data['size']}")
        elif self.sample_ratio:
            print(f"\n[SAMPLING] Ratio {self.sample_ratio} ({self.sample_ratio*100:.1f}%) applied")
            total_sampled = 0
            for task_name, task_data in self.benchmark['tasks'].items():
                original_size = len(task_data['data'])
                sample_size = max(1, int(original_size * self.sample_ratio))

                # Random sampling with fixed seed for reproducibility
                np.random.seed(42)
                indices = np.random.choice(original_size, sample_size, replace=False)
                task_data['data'] = [task_data['data'][i] for i in sorted(indices)]
                task_data['size'] = len(task_data['data'])
                total_sampled += task_data['size']
                print(f"  - {task_name}: {original_size} -> {task_data['size']} ({task_data['size']/original_size*100:.1f}%)")
            print(f"  - Total: {self.benchmark['benchmark_info']['total_size']} -> {total_sampled} ({total_sampled/self.benchmark['benchmark_info']['total_size']*100:.1f}%)")

    def setup_prompts(self):
        """프롬프트 템플릿 설정"""
        self.prompts = {
            'classification': {
                'system': "당신은 한국 고전 문헌 전문가입니다. 주어진 한문 텍스트의 문체를 정확하게 분류하세요.",
                'user': """다음 한문 텍스트의 문체를 분류하세요.

가능한 문체: 賦(부), 詩(시), 疑(의), 義(의), 策(책), 論(논), 表(표), 箋(전), 講(강), 頌(송), 箴(잠), 詔(조), 銘(명), 詩義, 禮義, 易義, 書義, 制(제), 擬(의)

텍스트: {input}

문체 (한 단어로만 답하세요):"""
            },
            'retrieval': {
                'system': "당신은 한국 고전 문헌 전문가입니다. 주어진 문장의 출처를 정확하게 식별하세요.",
                'user': """다음 한문 문장은 어느 책에서 나온 것인지 식별하세요.

가능한 책: 論語(논어), 孟子(맹자), 大學(대학), 中庸(중용)

문장: {input}

출처 (책 이름만 답하세요):"""
            },
            'punctuation': {
                'system': "당신은 한국 고전 문헌 전문가입니다. 구두점이 없는 한문(백문)에 적절한 구두점을 추가하세요.",
                'user': """다음 백문(구두점이 없는 한문)에 적절한 구두점을 추가하세요.

백문: {input}

구두점을 추가한 문장:"""
            },
            'nli': {
                'system': "당신은 한국 고전 문헌 전문가입니다. 두 문장 간의 논리적 관계를 판단하세요.",
                'user': """두 문장 간의 논리적 관계를 판단하세요.

전제(Premise): {premise}
가설(Hypothesis): {hypothesis}

관계:
- entailment: 전제가 참이면 가설도 반드시 참
- contradiction: 전제와 가설이 모순
- neutral: 관계를 알 수 없음

답변 (entailment, contradiction, neutral 중 하나만):"""
            },
            'translation': {
                'system': "당신은 한국 고전 문헌 번역 전문가입니다.",
                'user': """다음 텍스트를 {target_lang}(으)로 번역하세요.

원문 ({source_lang}): {input}

번역:"""
            }
        }

    def format_prompt(self, task: str, data: Dict) -> Tuple[str, str]:
        """프롬프트 포맷팅"""
        if task == 'nli':
            user_prompt = self.prompts[task]['user'].format(
                premise=data['premise'],
                hypothesis=data['hypothesis']
            )
        elif task == 'translation':
            # 언어 이름 매핑
            lang_names = {
                'classical_chinese': '한문',
                'korean': '한국어',
                'english': '영어'
            }
            user_prompt = self.prompts[task]['user'].format(
                source_lang=lang_names.get(data['source_lang'], data['source_lang']),
                target_lang=lang_names.get(data['target_lang'], data['target_lang']),
                input=data['source_text']
            )
        else:
            user_prompt = self.prompts[task]['user'].format(input=data['input'])

        return self.prompts[task]['system'], user_prompt

    def evaluate_classification(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """분류 태스크 평가"""
        # 정규화: 공백 제거, 소괄호 내용 제거
        def normalize(text):
            text = re.sub(r'\([^)]*\)', '', text)  # (부) 같은 표현 제거
            text = text.strip()
            return text

        preds_normalized = [normalize(p) for p in predictions]
        truths_normalized = [normalize(t) for t in ground_truths]

        # 디버그 출력 (소량 데이터일 때만)
        if len(predictions) <= 5:
            for i, (pred, truth) in enumerate(zip(preds_normalized, truths_normalized)):
                print(f"    {i+1}. Pred: '{pred}' vs Truth: '{truth}' -> {'✓' if pred == truth else '✗'}")

        accuracy = accuracy_score(truths_normalized, preds_normalized)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            truths_normalized, preds_normalized, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(predictions)
        }

    def evaluate_retrieval(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """검색 태스크 평가"""
        correct = 0

        for pred, truth in zip(predictions, ground_truths):
            # 책 이름 추출 (앞부분만)
            pred_book = pred.strip().split('-')[0].strip()
            truth_book = truth.strip().split('-')[0].strip()

            # 괄호 내용 먼저 제거, 그 다음 한국어→한자 정규화
            pred_book = re.sub(r'\([^)]*\)', '', pred_book).strip()
            truth_book = re.sub(r'\([^)]*\)', '', truth_book).strip()

            # 유니코드 정규화 (CJK Compatibility Ideographs → 표준 한자)
            pred_book = unicodedata.normalize('NFKC', pred_book)
            truth_book = unicodedata.normalize('NFKC', truth_book)

            # 한국어→한자 정규화
            pred_book = pred_book.replace('논어', '論語').replace('맹자', '孟子').replace('대학', '大學').replace('중용', '中庸')
            truth_book = truth_book.replace('논어', '論語').replace('맹자', '孟子').replace('대학', '大學').replace('중용', '中庸')

            # 부분 매칭도 허용
            match_result = (pred_book in truth_book or truth_book in pred_book)
            if match_result:
                correct += 1

            # 디버그 출력 (소량 데이터일 때만)
            if len(predictions) <= 5:
                item_num = len([p for p, t in zip(predictions, ground_truths) if p and t])
                print(f"    {item_num}. Pred: '{pred_book}' vs Truth: '{truth_book}' -> {'✓' if match_result else '✗'}")

        accuracy = correct / len(predictions) if predictions else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'num_samples': len(predictions)
        }

    def evaluate_punctuation(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """구두점 태스크 평가"""
        # Character-level F1
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        for pred, truth in zip(predictions, ground_truths):
            # 문자 단위 비교
            pred_chars = set(pred)
            truth_chars = set(truth)

            if len(pred_chars) == 0:
                continue

            intersection = pred_chars & truth_chars

            precision = len(intersection) / len(pred_chars) if pred_chars else 0
            recall = len(intersection) / len(truth_chars) if truth_chars else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n = len(predictions)

        # ROUGE score도 계산
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = [scorer.score(truth, pred) for pred, truth in zip(predictions, ground_truths)]

        rouge1_f1 = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
        rouge2_f1 = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
        rougeL_f1 = np.mean([s['rougeL'].fmeasure for s in rouge_scores])

        return {
            'char_precision': total_precision / n if n > 0 else 0,
            'char_recall': total_recall / n if n > 0 else 0,
            'char_f1': total_f1 / n if n > 0 else 0,
            'rouge1_f1': rouge1_f1,
            'rouge2_f1': rouge2_f1,
            'rougeL_f1': rougeL_f1,
            'num_samples': n
        }

    def evaluate_nli(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """NLI 태스크 평가"""
        # 정규화
        def normalize(text):
            text = text.lower().strip()
            # 한글도 처리
            mapping = {
                '함의': 'entailment',
                '모순': 'contradiction',
                '중립': 'neutral',
                '무관': 'neutral'
            }
            for k, v in mapping.items():
                text = text.replace(k, v)
            return text

        preds_normalized = [normalize(p) for p in predictions]
        truths_normalized = [normalize(t) for t in ground_truths]

        accuracy = accuracy_score(truths_normalized, preds_normalized)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            truths_normalized, preds_normalized, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(predictions)
        }

    def evaluate_translation(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """번역 태스크 평가"""
        bleu_scores = []
        rouge_scores_list = []

        # BLEU 계산
        smoothing = SmoothingFunction().method1

        for pred, truth in zip(predictions, ground_truths):
            # BLEU (character-level for Korean/Chinese)
            pred_chars = list(pred)
            truth_chars = list(truth)

            if len(pred_chars) > 0 and len(truth_chars) > 0:
                bleu = sentence_bleu([truth_chars], pred_chars, smoothing_function=smoothing)
                bleu_scores.append(bleu)

        # ROUGE 계산
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        for pred, truth in zip(predictions, ground_truths):
            score = scorer.score(truth, pred)
            rouge_scores_list.append(score)

        rouge1_f1 = np.mean([s['rouge1'].fmeasure for s in rouge_scores_list])
        rouge2_f1 = np.mean([s['rouge2'].fmeasure for s in rouge_scores_list])
        rougeL_f1 = np.mean([s['rougeL'].fmeasure for s in rouge_scores_list])

        return {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'rouge1_f1': rouge1_f1,
            'rouge2_f1': rouge2_f1,
            'rougeL_f1': rougeL_f1,
            'num_samples': len(predictions)
        }

    def evaluate_task(self, task_name: str, predictions: List[str], task_data: Dict) -> Dict:
        """단일 태스크 평가"""
        # Ground truth 추출
        if task_name == 'classification':
            ground_truths = [item['label'] for item in task_data['data']]
        elif task_name == 'retrieval':
            ground_truths = [item['answer'] for item in task_data['data']]
        elif task_name == 'punctuation':
            ground_truths = [item['answer'] for item in task_data['data']]
        elif task_name == 'nli':
            ground_truths = [item['label'] for item in task_data['data']]
        elif task_name == 'translation':
            ground_truths = [item['target_text'] for item in task_data['data']]
        else:
            raise ValueError(f"Unknown task: {task_name}")

        # 평가
        if task_name == 'classification':
            return self.evaluate_classification(predictions, ground_truths)
        elif task_name == 'retrieval':
            return self.evaluate_retrieval(predictions, ground_truths)
        elif task_name == 'punctuation':
            return self.evaluate_punctuation(predictions, ground_truths)
        elif task_name == 'nli':
            return self.evaluate_nli(predictions, ground_truths)
        elif task_name == 'translation':
            return self.evaluate_translation(predictions, ground_truths)

    def run_evaluation(self, model, model_name: str) -> Dict:
        """전체 벤치마크 평가 실행"""
        print(f"\n{'='*70}")
        print(f"[START] Model evaluation: {model_name}")
        print(f"[TEMPERATURE] {self.temperature}")
        print(f"{'='*70}\n")

        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'temperature': self.temperature,
            'benchmark_version': self.benchmark['benchmark_info']['version'],
            'tasks': {}
        }

        for task_name, task_data in self.benchmark['tasks'].items():
            print(f"\n[{task_name.upper()}] Evaluating {task_data['size']} samples...")

            predictions = []
            sample_outputs = []  # For appendix

            for idx, item in enumerate(tqdm(task_data['data'], desc=f"  Processing {task_name}")):
                # 프롬프트 생성
                system_prompt, user_prompt = self.format_prompt(task_name, item)

                # 모델 추론
                try:
                    prediction = model.generate(system_prompt, user_prompt)
                    if not prediction or prediction.strip() == "":
                        print(f"  [WARNING] Empty prediction for item {len(predictions)+1}")
                    predictions.append(prediction)

                    # Save sample outputs for appendix
                    if self.save_samples and idx < self.num_samples_to_save:
                        sample_output = {
                            'sample_id': idx,
                            'system_prompt': system_prompt,
                            'user_prompt': user_prompt,
                            'prediction': prediction,
                            'ground_truth': self._get_ground_truth(task_name, item)
                        }
                        sample_outputs.append(sample_output)

                except Exception as e:
                    print(f"  [ERROR] Model generation error: {e}")
                    predictions.append("")

                # API 호출 제한 대응
                if self.model_type == 'api':
                    time.sleep(1.0)  # Rate limiting - increased for GPT-4

            # 평가
            metrics = self.evaluate_task(task_name, predictions, task_data)

            results['tasks'][task_name] = {
                'metrics': metrics,
                'predictions': predictions[:10],  # 처음 10개만 저장
                'sample_outputs': sample_outputs if self.save_samples else []
            }

            print(f"  ✓ 완료")
            self.print_task_results(task_name, metrics)

        # 결과 저장
        self.save_results(results, model_name)

        return results

    def _get_ground_truth(self, task_name: str, item: Dict) -> str:
        """Extract ground truth for a given task and item"""
        if task_name == 'classification':
            return item['label']
        elif task_name == 'retrieval':
            return item['answer']
        elif task_name == 'punctuation':
            return item['answer']
        elif task_name == 'nli':
            return item['label']
        elif task_name == 'translation':
            return item['target_text']
        else:
            return ""

    def print_task_results(self, task_name: str, metrics: Dict):
        """태스크 결과 출력"""
        print(f"\n   {task_name.upper()} 결과:")

        if task_name == 'classification':
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - F1 Score: {metrics['f1']:.4f}")

        elif task_name == 'retrieval':
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - Correct: {metrics['correct']}/{metrics['num_samples']}")

        elif task_name == 'punctuation':
            print(f"    - Character F1: {metrics['char_f1']:.4f}")
            print(f"    - ROUGE-L F1: {metrics['rougeL_f1']:.4f}")

        elif task_name == 'nli':
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - F1 Score: {metrics['f1']:.4f}")

        elif task_name == 'translation':
            print(f"    - BLEU: {metrics['bleu']:.4f}")
            print(f"    - ROUGE-L F1: {metrics['rougeL_f1']:.4f}")

    def save_results(self, results: Dict, model_name: str):
        """결과 저장"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        temperature = results.get('temperature', 0.0)

        # 모델명에서 슬래시를 언더스코어로 변경 (파일명에 사용 가능하도록)
        safe_model_name = model_name.replace('/', '_')

        # JSON 저장 (temperature 포함)
        json_path = self.output_dir / f"results_{safe_model_name}_temp{temperature:.1f}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[SAVE] Results saved to: {json_path}")

        # CSV 요약 저장
        summary_data = []
        for task_name, task_results in results['tasks'].items():
            row = {
                'model': model_name,
                'task': task_name,
                **task_results['metrics']
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / f"summary_{safe_model_name}_temp{temperature:.1f}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"[SAVE] Summary saved to: {csv_path}")


# ============================================================================
# 모델 래퍼 클래스
# ============================================================================

class BaseModelWrapper:
    """기본 모델 래퍼"""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIWrapper(BaseModelWrapper):
    """OpenAI API 래퍼 (GPT-4, GPT-3.5 등)"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, temperature: float = 0.0):
        import openai
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            content = response.choices[0].message.content
            if content is None:
                print(f"[WARNING] Empty response from {self.model_name}")
                return ""
            return content.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI API Error: {e}")
            print(f"   Model: {self.model_name}")
            print(f"   System: {system_prompt[:50]}...")
            print(f"   User: {user_prompt[:50]}...")
            return ""


class AnthropicWrapper(BaseModelWrapper):
    """Anthropic Claude API 래퍼"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, temperature: float = 0.0):
        import anthropic
        self.model_name = model_name
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""


class HuggingFaceWrapper(BaseModelWrapper):
    """HuggingFace 모델 래퍼 (Llama, Qwen, EXAONE 등)"""

    def __init__(self, model_name: str, device: str = "cuda", use_auth_token: bool = True, temperature: float = 0.0):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.device = device
        self.model_name = model_name
        self.temperature = temperature

        print(f"Loading {model_name}...")
        # HuggingFace token 사용 (gated models를 위해)
        # use_auth_token is deprecated, use token instead
        token = True if use_auth_token else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            token=token,
            trust_remote_code=True
        )
        print(f"✓ Model loaded")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Chat template 사용
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback: simple concatenation
            prompt = f"{system_prompt}\n\n{user_prompt}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Temperature handling
        if self.temperature > 0:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=1.0,  # dummy value when do_sample=False
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated tokens (excluding input)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


class TonguWrapper(BaseModelWrapper):
    """Tongu 모델 래퍼 - SCUT-DLVCLab/TongGu-7B-Instruct"""

    def __init__(self, model_path: str = "SCUT-DLVCLab/TongGu-7B-Instruct", device: str = "cuda", temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.temperature = temperature
        print(f"[WARNING] Tongu wrapper not implemented: {model_path}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Tongu의 프롬프트 형식: system_message + "\n<用户> " + query + "\n<通古> "
        # system_prompt는 무시하고 Tongu의 기본 시스템 메시지 사용
        prompt = f"{self.system_message}\n<用户> {user_prompt}\n<通古> "

        try:
            inputs = self.tokenizer(prompt, return_tensors='pt')

            if self.temperature > 0:
                generate_ids = self.model.generate(
                    inputs.input_ids.to(self.device),
                    max_new_tokens=500,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            else:
                generate_ids = self.model.generate(
                    inputs.input_ids.to(self.device),
                    max_new_tokens=500,
                    temperature=1.0,
                    do_sample=False
                )

            generate_text = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0][len(prompt):]

            return generate_text.strip()
        except Exception as e:
            print(f" Tongu generation error: {e}")
            return ""


class GwenBertWrapper(BaseModelWrapper):
    """GwenBert 모델 래퍼 - ethanyt/guwenbert-base

    Note: GwenBERT는 BERT 기반 인코더 모델로 생성 태스크에는 적합하지 않습니다.
    주로 분류, 임베딩 등의 태스크에 사용됩니다.
    이 래퍼는 제한적인 기능만 제공합니다.
    """

    def __init__(self, model_path: str = "ethanyt/guwenbert-base", device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.model_path = model_path
        print(f"[WARNING] GwenBert wrapper not implemented: {model_path}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        GwenBERT는 생성 모델이 아니므로 실제 생성 불가능
        임베딩만 추출 가능하여 벤치마크 평가에 적합하지 않음
        """
        print(f"  GwenBERT는 생성 태스크를 지원하지 않습니다.")
        return "[GwenBERT는 생성 모델이 아닙니다]"


# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    import argparse

    # Load config for default values
    config = None
    if CONFIG_AVAILABLE:
        try:
            config = Config()
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")

    # Set defaults from config or fallback values (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # experiments/exp5 -> notebook -> korean_eda

    default_benchmark = config.get_benchmark_path() if config else \
        str(project_root / 'benchmark' / 'kls_bench' / 'kls_bench_full.json')
    default_output = config.get_output_dir() if config else \
        str(project_root / 'results')

    parser = argparse.ArgumentParser(description='KLSBench Evaluation')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (default: auto-detect)')
    parser.add_argument('--benchmark', type=str,
                       default=default_benchmark,
                       help='Benchmark JSON file path')
    parser.add_argument('--output', type=str,
                       default=default_output,
                       help='Output directory for results')
    parser.add_argument('--model-type', type=str, choices=['api', 'opensource', 'supervised'],
                       default='api', help='Model type')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name (e.g., gpt-4, claude-3-5-sonnet, meta-llama/Llama-3.1-8B)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (for API models)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per task (for testing)')
    parser.add_argument('--sample-ratio', type=float, default=None,
                       help='Sampling ratio (0.0~1.0, e.g., 0.3=30%%)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (0.0~1.0, default: 0.0)')
    parser.add_argument('--save-samples', action='store_true', default=True,
                       help='Save sample outputs for appendix (default: True)')
    parser.add_argument('--num-samples-to-save', type=int, default=5,
                       help='Number of samples to save per task (default: 5)')

    args = parser.parse_args()

    # Reload config if custom path provided
    if args.config and CONFIG_AVAILABLE:
        config = Config(args.config)

    # Evaluator 초기화
    evaluator = KLSBenchEvaluator(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        model_type=args.model_type,
        max_samples_per_task=args.max_samples,
        sample_ratio=args.sample_ratio,
        temperature=args.temperature,
        save_samples=args.save_samples,
        num_samples_to_save=args.num_samples_to_save
    )

    # 모델 초기화
    if args.model_type == 'api':
        if 'gpt' in args.model_name.lower():
            model = OpenAIWrapper(model_name=args.model_name, api_key=args.api_key, temperature=args.temperature)
        elif 'claude' in args.model_name.lower():
            model = AnthropicWrapper(model_name=args.model_name, api_key=args.api_key, temperature=args.temperature)
        else:
            raise ValueError(f"Unknown API model: {args.model_name}")

    elif args.model_type == 'opensource':
        model = HuggingFaceWrapper(model_name=args.model_name, temperature=args.temperature)

    elif args.model_type == 'supervised':
        if 'tongu' in args.model_name.lower():
            model = TonguWrapper(model_path=args.model_name, temperature=args.temperature)
        elif 'gwenbert' in args.model_name.lower():
            model = GwenBertWrapper(model_path=args.model_name)
        else:
            raise ValueError(f"Unknown supervised model: {args.model_name}")

    # 평가 실행
    results = evaluator.run_evaluation(model, args.model_name)

    print("\n" + "="*70)
    print("[COMPLETE] Evaluation finished")
    print("="*70)


if __name__ == "__main__":
    main()
