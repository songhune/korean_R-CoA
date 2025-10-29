"""
K-ClassicBench Evaluation Framework
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
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class KClassicBenchEvaluator:
    """K-ClassicBench 벤치마크 평가기"""

    def __init__(self,
                 benchmark_path: str,
                 output_dir: str,
                 model_type: str = "api",  # "api", "opensource", "supervised"
                 max_samples_per_task: Optional[int] = None,
                 sample_ratio: Optional[float] = None):
        """
        Args:
            benchmark_path: 벤치마크 JSON 파일 경로
            output_dir: 결과 저장 디렉토리
            model_type: 모델 타입 (api/opensource/supervised)
            max_samples_per_task: 태스크당 최대 샘플 수 (None이면 전체)
            sample_ratio: 샘플링 비율 (0.0~1.0, None이면 전체)
        """
        self.benchmark_path = Path(benchmark_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.max_samples_per_task = max_samples_per_task
        self.sample_ratio = sample_ratio

        # 벤치마크 로드
        self.load_benchmark()

        # 프롬프트 템플릿
        self.setup_prompts()

    def load_benchmark(self):
        """벤치마크 데이터 로드"""
        print(f"📂 벤치마크 로딩: {self.benchmark_path}")
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)

        print(f"  ✓ {self.benchmark['benchmark_info']['name']}")
        print(f"  ✓ 총 {self.benchmark['benchmark_info']['total_size']:,}개 항목")
        print(f"  ✓ {len(self.benchmark['tasks'])}개 태스크")

        # 샘플 제한 적용 (우선순위: max_samples > sample_ratio)
        if self.max_samples_per_task:
            print(f"\n⚠️  각 태스크당 최대 {self.max_samples_per_task}개 샘플로 제한")
            for task_name, task_data in self.benchmark['tasks'].items():
                original_size = len(task_data['data'])
                task_data['data'] = task_data['data'][:self.max_samples_per_task]
                task_data['size'] = len(task_data['data'])
                print(f"  - {task_name}: {original_size} → {task_data['size']}")
        elif self.sample_ratio:
            print(f"\n📊 샘플링 비율 {self.sample_ratio} ({self.sample_ratio*100:.1f}%) 적용")
            total_sampled = 0
            for task_name, task_data in self.benchmark['tasks'].items():
                original_size = len(task_data['data'])
                sample_size = max(1, int(original_size * self.sample_ratio))

                # 랜덤 샘플링 (재현성을 위해 seed 고정)
                np.random.seed(42)
                indices = np.random.choice(original_size, sample_size, replace=False)
                task_data['data'] = [task_data['data'][i] for i in sorted(indices)]
                task_data['size'] = len(task_data['data'])
                total_sampled += task_data['size']
                print(f"  - {task_name}: {original_size} → {task_data['size']} ({task_data['size']/original_size*100:.1f}%)")
            print(f"  - 총계: {self.benchmark['benchmark_info']['total_size']} → {total_sampled} ({total_sampled/self.benchmark['benchmark_info']['total_size']*100:.1f}%)")

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
        print(f"🚀 모델 평가 시작: {model_name}")
        print(f"{'='*70}\n")

        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'benchmark_version': self.benchmark['benchmark_info']['version'],
            'tasks': {}
        }

        for task_name, task_data in self.benchmark['tasks'].items():
            print(f"\n📊 [{task_name.upper()}] 평가 중... ({task_data['size']}개 샘플)")

            predictions = []

            for item in tqdm(task_data['data'], desc=f"  Processing {task_name}"):
                # 프롬프트 생성
                system_prompt, user_prompt = self.format_prompt(task_name, item)

                # 모델 추론
                try:
                    prediction = model.generate(system_prompt, user_prompt)
                    if not prediction or prediction.strip() == "":
                        print(f"  ⚠️  Empty prediction for item {len(predictions)+1}")
                    predictions.append(prediction)
                except Exception as e:
                    print(f"  ❌ Model generation error: {e}")
                    predictions.append("")

                # API 호출 제한 대응
                if self.model_type == 'api':
                    time.sleep(1.0)  # Rate limiting - increased for GPT-4

            # 평가
            metrics = self.evaluate_task(task_name, predictions, task_data)

            results['tasks'][task_name] = {
                'metrics': metrics,
                'predictions': predictions[:10]  # 처음 10개만 저장
            }

            print(f"  ✓ 완료")
            self.print_task_results(task_name, metrics)

        # 결과 저장
        self.save_results(results, model_name)

        return results

    def print_task_results(self, task_name: str, metrics: Dict):
        """태스크 결과 출력"""
        print(f"\n  📈 {task_name.upper()} 결과:")

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

        # JSON 저장
        json_path = self.output_dir / f"results_{model_name}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 결과 저장: {json_path}")

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
        csv_path = self.output_dir / f"summary_{model_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"💾 요약 저장: {csv_path}")


# ============================================================================
# 모델 래퍼 클래스
# ============================================================================

class BaseModelWrapper:
    """기본 모델 래퍼"""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIWrapper(BaseModelWrapper):
    """OpenAI API 래퍼 (GPT-4, GPT-3.5 등)"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            content = response.choices[0].message.content
            if content is None:
                print(f"⚠️  Warning: Empty response from {self.model_name}")
                return ""
            return content.strip()
        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            print(f"   Model: {self.model_name}")
            print(f"   System: {system_prompt[:50]}...")
            print(f"   User: {user_prompt[:50]}...")
            return ""


class AnthropicWrapper(BaseModelWrapper):
    """Anthropic Claude API 래퍼"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.0,
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

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
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

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        response = response[len(prompt):].strip()

        return response


class TonguWrapper(BaseModelWrapper):
    """Tongu 모델 래퍼"""

    def __init__(self, model_path: str):
        # Tongu 모델 로드 로직
        # TODO: 실제 모델 로드 구현
        self.model_path = model_path
        print(f"⚠️  Tongu wrapper - 구현 필요: {model_path}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # TODO: Tongu 모델 추론 구현
        return ""


class GwenBertWrapper(BaseModelWrapper):
    """GwenBert 모델 래퍼"""

    def __init__(self, model_path: str):
        # GwenBert 모델 로드 로직
        # TODO: 실제 모델 로드 구현
        self.model_path = model_path
        print(f"⚠️  GwenBert wrapper - 구현 필요: {model_path}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # TODO: GwenBert 모델 추론 구현
        return ""


# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='K-ClassicBench Evaluation')
    parser.add_argument('--benchmark', type=str,
                       default='/Users/songhune/Workspace/korean_eda/benchmark/k_classic_bench/k_classic_bench_full.json',
                       help='벤치마크 JSON 파일 경로')
    parser.add_argument('--output', type=str,
                       default='/Users/songhune/Workspace/korean_eda/benchmark/results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--model-type', type=str, choices=['api', 'opensource', 'supervised'],
                       default='api', help='모델 타입')
    parser.add_argument('--model-name', type=str, required=True,
                       help='모델 이름 (예: gpt-4, claude-3-5-sonnet, meta-llama/Llama-3.1-8B)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API 키 (API 모델 사용시)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='태스크당 최대 샘플 수 (테스트용)')
    parser.add_argument('--sample-ratio', type=float, default=None,
                       help='샘플링 비율 (0.0~1.0, 예: 0.3=30%%)')

    args = parser.parse_args()

    # Evaluator 초기화
    evaluator = KClassicBenchEvaluator(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        model_type=args.model_type,
        max_samples_per_task=args.max_samples,
        sample_ratio=args.sample_ratio
    )

    # 모델 초기화
    if args.model_type == 'api':
        if 'gpt' in args.model_name.lower():
            model = OpenAIWrapper(model_name=args.model_name, api_key=args.api_key)
        elif 'claude' in args.model_name.lower():
            model = AnthropicWrapper(model_name=args.model_name, api_key=args.api_key)
        else:
            raise ValueError(f"Unknown API model: {args.model_name}")

    elif args.model_type == 'opensource':
        model = HuggingFaceWrapper(model_name=args.model_name)

    elif args.model_type == 'supervised':
        if 'tongu' in args.model_name.lower():
            model = TonguWrapper(model_path=args.model_name)
        elif 'gwenbert' in args.model_name.lower():
            model = GwenBertWrapper(model_path=args.model_name)
        else:
            raise ValueError(f"Unknown supervised model: {args.model_name}")

    # 평가 실행
    results = evaluator.run_evaluation(model, args.model_name)

    print("\n" + "="*70)
    print("✅ 평가 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
