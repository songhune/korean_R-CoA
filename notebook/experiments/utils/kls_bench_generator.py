"""
KLSBench: Korean Classical Literature Understanding Benchmark Generator
한국 고전 문헌 이해 벤치마크 생성기

C3Bench를 참고하여 5가지 핵심 태스크로 구성:
1. Classification (분류): 문체 분류
2. Retrieval (검색): 출처 식별
3. Punctuation (구두점): 백문에 구두점 복원
4. NLI/STS: 자연언어추론 및 의미 유사도
5. Translation (번역): 한문-한글-영문 번역

Target: 총 10,000개 항목 (각 태스크당 2,000개)
"""

import pandas as pd
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

# 랜덤 시드 고정
random.seed(42)
np.random.seed(42)


class KLSBenchGenerator:
    """한국 고전 문헌 벤치마크 생성기"""

    def __init__(self,
                 translated_csv_path: str,
                 external_csv_path: str,
                 nli_examples_path: str,
                 output_dir: str):
        """
        Args:
            translated_csv_path: 과거시험 번역 데이터 경로
            external_csv_path: 사서 원문 데이터 경로
            nli_examples_path: NLI 예시 데이터 경로
            output_dir: 출력 디렉토리
        """
        self.translated_csv_path = translated_csv_path
        self.external_csv_path = external_csv_path
        self.nli_examples_path = nli_examples_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 로드
        self.load_data()

    def load_data(self):
        """데이터 로드"""
        print("📂 데이터 로딩 중...")

        # 과거시험 데이터
        self.translated_df = pd.read_csv(self.translated_csv_path)
        print(f"  - 과거시험 데이터: {len(self.translated_df)} 항목")

        # 사서 데이터
        self.external_df = pd.read_csv(self.external_csv_path)
        print(f"  - 사서 데이터: {len(self.external_df)} 항목")

        # NLI 예시
        with open(self.nli_examples_path, 'r', encoding='utf-8') as f:
            nli_data = json.load(f)
            self.nli_examples = nli_data.get('examples', [])
        print(f"  - NLI 예시: {len(self.nli_examples)} 항목")

        # 데이터 전처리
        self.preprocess_data()

    def preprocess_data(self):
        """데이터 전처리"""
        print("\n🔧 데이터 전처리 중...")

        # 과거시험 데이터: 빈 값 제거
        self.translated_df = self.translated_df.dropna(subset=['category'])
        self.translated_df = self.translated_df[
            (self.translated_df['abstract'].notna()) |
            (self.translated_df['content'].notna())
        ]

        # 사서 데이터: 빈 값 제거
        self.external_df = self.external_df.dropna(subset=['Original', 'Book'])

        print(f"  - 전처리 후 과거시험 데이터: {len(self.translated_df)} 항목")
        print(f"  - 전처리 후 사서 데이터: {len(self.external_df)} 항목")

    def generate_classification_task(self, target_size: int = 2000) -> List[Dict]:
        """
        Task 1: 분류 (Classification)
        문체(賦/詩/疑/義) 분류 태스크 생성
        """
        print(f"\n📋 [1/5] 분류(Classification) 태스크 생성 중... (목표: {target_size}개)")

        task_data = []

        # 카테고리별 샘플링
        categories = self.translated_df['category'].unique()
        category_counts = self.translated_df['category'].value_counts()

        print(f"  카테고리: {list(categories)}")
        print(f"  분포: {dict(category_counts)}")

        # 균등 샘플링을 위한 계산
        samples_per_category = target_size // len(categories)

        for category in categories:
            category_df = self.translated_df[self.translated_df['category'] == category]

            # 샘플 수 결정
            n_samples = min(samples_per_category, len(category_df))
            sampled_df = category_df.sample(n=n_samples, random_state=42)

            for idx, row in sampled_df.iterrows():
                # 입력 텍스트: abstract 또는 content 사용
                input_text = row['abstract'] if pd.notna(row['abstract']) and row['abstract'].strip() else row['content']

                if pd.notna(input_text) and input_text.strip():
                    task_data.append({
                        'task': 'classification',
                        'id': f"cls_{len(task_data)+1:04d}",
                        'input': input_text.strip(),
                        'label': category,
                        'question_id': row.get('question_id', ''),
                        'metadata': {
                            'has_korean': pd.notna(row.get('abstract_ko')) or pd.notna(row.get('content_ko')),
                            'has_english': pd.notna(row.get('abstract_en')) or pd.notna(row.get('content_en'))
                        }
                    })

        print(f"  ✓ 생성 완료: {len(task_data)} 항목")
        return task_data[:target_size]

    def generate_retrieval_task(self, target_size: int = 2000) -> List[Dict]:
        """
        Task 2: 검색 (Retrieval)
        주어진 문장의 출처(Book/Chapter) 식별 태스크 생성
        """
        print(f"\n🔍 [2/5] 검색(Retrieval) 태스크 생성 중... (목표: {target_size}개)")

        task_data = []

        # 사서 데이터에서 추출
        available_books = self.external_df['Book'].unique()
        print(f"  책 목록: {list(available_books)}")

        # 책별 균등 샘플링
        samples_per_book = target_size // len(available_books)

        for book in available_books:
            book_df = self.external_df[self.external_df['Book'] == book]

            # 샘플 수 결정
            n_samples = min(samples_per_book, len(book_df))
            sampled_df = book_df.sample(n=n_samples, random_state=42)

            for idx, row in sampled_df.iterrows():
                original_text = row['Original']

                if pd.notna(original_text) and original_text.strip():
                    # 정답: Book + Chapter
                    answer = f"{row['Book']}"
                    if pd.notna(row.get('Chapter')):
                        answer += f" - {row['Chapter']}"

                    task_data.append({
                        'task': 'retrieval',
                        'id': f"ret_{len(task_data)+1:04d}",
                        'input': original_text.strip(),
                        'answer': answer,
                        'book': row['Book'],
                        'chapter': row.get('Chapter', ''),
                        'volume': row.get('Volume', ''),
                        'metadata': {
                            'has_translation': pd.notna(row.get('Original_trans')),
                            'has_comment': pd.notna(row.get('Comment'))
                        }
                    })

        print(f"  ✓ 생성 완료: {len(task_data)} 항목")
        return task_data[:target_size]

    def generate_punctuation_task(self, target_size: int = 2000) -> List[Dict]:
        """
        Task 3: 구두점 찍기 (Punctuation)
        백문(구두점 없는 한문)에 구두점 복원 태스크 생성

        Original (구두점 없는 한문) → Original_quotation (구두점 있는 한문)
        Comment (구두점 없는 주석) → Comment_quotation (구두점 있는 주석)
        """
        print(f"\n✏️  [3/5] 구두점(Punctuation) 태스크 생성 중... (목표: {target_size}개)")

        task_data = []

        # 1) 사서 데이터: Original (백문) → Original_quotation (구두점 있는 한문)
        external_with_quotation = self.external_df[
            (self.external_df['Original'].notna()) &
            (self.external_df['Original_quotation'].notna())
        ]

        for idx, row in external_with_quotation.iterrows():
            if len(task_data) >= target_size * 0.5:  # 전체의 50%
                break

            original_text = row['Original'].strip()
            quotation_text = row['Original_quotation'].strip()

            # 백문과 구두점본이 다른 경우만 사용 (구두점이 실제로 추가된 경우)
            if original_text and quotation_text and original_text != quotation_text:
                if len(original_text) > 10:  # 최소 길이 제한
                    task_data.append({
                        'task': 'punctuation',
                        'id': f"pun_{len(task_data)+1:04d}",
                        'input': original_text,
                        'answer': quotation_text,
                        'language': 'classical_chinese',
                        'text_type': 'original',
                        'source': row.get('Book', ''),
                        'metadata': {
                            'chapter': row.get('Chapter', ''),
                            'korean_translation': row.get('Original_trans', ''),
                            'original_length': len(original_text),
                            'punctuated_length': len(quotation_text)
                        }
                    })

        # 2) 사서 데이터: Comment (백문 주석) → Comment_quotation (구두점 있는 주석)
        external_with_comment_quotation = self.external_df[
            (self.external_df['Comment'].notna()) &
            (self.external_df['Comment_quotation'].notna())
        ]

        for idx, row in external_with_comment_quotation.iterrows():
            if len(task_data) >= target_size:
                break

            comment_text = row['Comment'].strip()
            comment_quotation_text = row['Comment_quotation'].strip()

            # 백문과 구두점본이 다른 경우만 사용
            if comment_text and comment_quotation_text and comment_text != comment_quotation_text:
                if len(comment_text) > 10:
                    task_data.append({
                        'task': 'punctuation',
                        'id': f"pun_{len(task_data)+1:04d}",
                        'input': comment_text,
                        'answer': comment_quotation_text,
                        'language': 'classical_chinese',
                        'text_type': 'comment',
                        'source': row.get('Book', ''),
                        'metadata': {
                            'chapter': row.get('Chapter', ''),
                            'korean_translation': row.get('Comment_trans', ''),
                            'original_length': len(comment_text),
                            'punctuated_length': len(comment_quotation_text)
                        }
                    })

        print(f"  ✓ 생성 완료: {len(task_data)} 항목")
        return task_data[:target_size]

    def generate_nli_sts_task(self, target_size: int = 2000) -> List[Dict]:
        """
        Task 4: NLI/STS (Natural Language Inference / Semantic Textual Similarity)
        자연언어추론 및 의미 유사도 태스크 생성
        """
        print(f"\n🧠 [4/5] NLI/STS 태스크 생성 중... (목표: {target_size}개)")

        task_data = []

        # 기존 NLI 예시를 템플릿으로 활용
        print(f"  기존 NLI 예시 {len(self.nli_examples)}개를 템플릿으로 활용")

        # 1) 기존 예시 추가 (15개)
        for example in self.nli_examples:
            task_data.append({
                'task': 'nli',
                'id': f"nli_{len(task_data)+1:04d}",
                'premise': example['premise'],
                'hypothesis': example['hypothesis'],
                'label': example['label'],
                'source': example.get('source', ''),
                'difficulty': example.get('difficulty', 'medium'),
                'category': example.get('category', ''),
                'explanation': example.get('explanation', '')
            })

        # 2) 사서 데이터로 NLI 쌍 생성
        # Entailment: 원문 → 번역 관계
        entailment_samples = self.external_df[
            self.external_df['Original_trans'].notna()
        ].sample(n=min(650, len(self.external_df)), random_state=42)

        for idx, row in entailment_samples.iterrows():
            if len(task_data) >= target_size:
                break

            task_data.append({
                'task': 'nli',
                'id': f"nli_{len(task_data)+1:04d}",
                'premise': row['Original'].strip(),
                'hypothesis': row['Original_trans'].strip(),
                'label': 'entailment',
                'source': row.get('Book', ''),
                'difficulty': 'easy',
                'category': 'translation_equivalence',
                'explanation': '원문과 번역이 의미적으로 동일'
            })

        # 3) 과거시험 데이터로 번역 관계 NLI 생성
        for idx, row in self.translated_df.iterrows():
            if len(task_data) >= target_size * 0.66:  # 전체의 66%까지
                break

            # 한문 → 한글 번역
            if pd.notna(row['abstract']) and pd.notna(row['abstract_ko']):
                task_data.append({
                    'task': 'nli',
                    'id': f"nli_{len(task_data)+1:04d}",
                    'premise': row['abstract'].strip(),
                    'hypothesis': row['abstract_ko'].strip(),
                    'label': 'entailment',
                    'source': '과거시험',
                    'difficulty': 'easy',
                    'category': 'translation_equivalence',
                    'explanation': '한문과 한글 번역이 의미적으로 동일'
                })

            if len(task_data) >= target_size:
                break

            # 한글 → 영문 번역
            if pd.notna(row['abstract_ko']) and pd.notna(row['abstract_en']):
                task_data.append({
                    'task': 'nli',
                    'id': f"nli_{len(task_data)+1:04d}",
                    'premise': row['abstract_ko'].strip(),
                    'hypothesis': row['abstract_en'].strip(),
                    'label': 'entailment',
                    'source': '과거시험',
                    'difficulty': 'medium',
                    'category': 'cross_lingual_entailment',
                    'explanation': '한글과 영어 번역이 의미적으로 동일'
                })

        # 4) Neutral/Contradiction 예시 생성 (개선된 휴리스틱)
        # Neutral: 다른 책의 문장끼리 매칭
        books = self.external_df['Book'].unique()
        neutral_target = int(target_size * 0.20)  # 전체의 20%

        # 책별 페어 생성
        book_pairs = []
        for i, book1 in enumerate(books):
            for book2 in books[i+1:]:
                book_pairs.append((book1, book2))

        for book1, book2 in book_pairs:
            if len([item for item in task_data if item['label'] == 'neutral']) >= neutral_target:
                break

            book1_samples = self.external_df[self.external_df['Book'] == book1].sample(
                n=min(100, len(self.external_df[self.external_df['Book'] == book1])), random_state=42
            )
            book2_samples = self.external_df[self.external_df['Book'] == book2].sample(
                n=min(100, len(self.external_df[self.external_df['Book'] == book2])), random_state=42
            )

            for (idx1, row1), (idx2, row2) in zip(book1_samples.iterrows(), book2_samples.iterrows()):
                if len([item for item in task_data if item['label'] == 'neutral']) >= neutral_target:
                    break

                if pd.notna(row1['Original_trans']) and pd.notna(row2['Original_trans']):
                    task_data.append({
                        'task': 'nli',
                        'id': f"nli_{len(task_data)+1:04d}",
                        'premise': row1['Original_trans'].strip(),
                        'hypothesis': row2['Original_trans'].strip(),
                        'label': 'neutral',
                        'source': f"{row1['Book']} vs {row2['Book']}",
                        'difficulty': 'medium',
                        'category': 'cross_text_relation',
                        'explanation': '다른 문헌의 문장으로 관계 불명'
                    })

        # 5) Contradiction 예시 생성
        # 부정문 생성 (간단한 휴리스틱)
        contradiction_target = int(target_size * 0.15)  # 전체의 15%

        # 과거시험 데이터에서 contradiction 쌍 생성
        # 다른 카테고리끼리 매칭 → contradiction 가능성 높음
        categories = self.translated_df['category'].unique()

        for cat1, cat2 in zip(categories[:-1], categories[1:]):
            if len([item for item in task_data if item['label'] == 'contradiction']) >= contradiction_target:
                break

            cat1_df = self.translated_df[
                (self.translated_df['category'] == cat1) &
                (self.translated_df['abstract_ko'].notna())
            ]
            cat2_df = self.translated_df[
                (self.translated_df['category'] == cat2) &
                (self.translated_df['abstract_ko'].notna())
            ]

            if len(cat1_df) == 0 or len(cat2_df) == 0:
                continue

            cat1_samples = cat1_df.sample(n=min(50, len(cat1_df)), random_state=42)
            cat2_samples = cat2_df.sample(n=min(50, len(cat2_df)), random_state=42)

            for (idx1, row1), (idx2, row2) in zip(cat1_samples.iterrows(), cat2_samples.iterrows()):
                if len([item for item in task_data if item['label'] == 'contradiction']) >= contradiction_target:
                    break

                # 부정 표현 추가로 contradiction 생성
                premise = row1['abstract_ko'].strip()
                hypothesis = f"{row2['abstract_ko'].strip()}이 아니다"  # 부정 추가

                task_data.append({
                    'task': 'nli',
                    'id': f"nli_{len(task_data)+1:04d}",
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label': 'contradiction',
                    'source': '과거시험 (생성)',
                    'difficulty': 'medium',
                    'category': 'negation_based',
                    'explanation': '부정 표현을 통한 모순 관계'
                })

        print(f"  ✓ 생성 완료: {len(task_data)} 항목")
        print(f"    - Label 분포: {Counter([item['label'] for item in task_data])}")
        return task_data[:target_size]

    def generate_translation_task(self, target_size: int = 2000) -> List[Dict]:
        """
        Task 5: 번역 (Translation)
        한문 ↔ 한글 ↔ 영문 번역 태스크 생성
        """
        print(f"\n🌐 [5/5] 번역(Translation) 태스크 생성 중... (목표: {target_size}개)")

        task_data = []

        # 1) 사서 데이터: 한문 → 한글
        external_with_trans = self.external_df[
            self.external_df['Original_trans'].notna()
        ].sample(n=min(800, len(self.external_df)), random_state=42)

        for idx, row in external_with_trans.iterrows():
            task_data.append({
                'task': 'translation',
                'id': f"trans_{len(task_data)+1:04d}",
                'source_text': row['Original'].strip(),
                'target_text': row['Original_trans'].strip(),
                'source_lang': 'classical_chinese',
                'target_lang': 'korean',
                'book': row.get('Book', ''),
                'metadata': {
                    'has_comment': pd.notna(row.get('Comment'))
                }
            })

        # 2) 과거시험 데이터: 한문 → 한글
        for idx, row in self.translated_df.iterrows():
            if len(task_data) >= target_size * 0.66:
                break

            # abstract: 한문 → 한글
            if pd.notna(row['abstract']) and pd.notna(row['abstract_ko']):
                task_data.append({
                    'task': 'translation',
                    'id': f"trans_{len(task_data)+1:04d}",
                    'source_text': row['abstract'].strip(),
                    'target_text': row['abstract_ko'].strip(),
                    'source_lang': 'classical_chinese',
                    'target_lang': 'korean',
                    'category': row.get('category', ''),
                    'metadata': {}
                })

            if len(task_data) >= target_size:
                break

            # content: 한문 → 한글
            if pd.notna(row['content']) and pd.notna(row['content_ko']):
                task_data.append({
                    'task': 'translation',
                    'id': f"trans_{len(task_data)+1:04d}",
                    'source_text': row['content'].strip(),
                    'target_text': row['content_ko'].strip(),
                    'source_lang': 'classical_chinese',
                    'target_lang': 'korean',
                    'category': row.get('category', ''),
                    'metadata': {}
                })

        # 3) 과거시험 데이터: 한글 → 영문
        for idx, row in self.translated_df.iterrows():
            if len(task_data) >= target_size:
                break

            if pd.notna(row['abstract_ko']) and pd.notna(row['abstract_en']):
                task_data.append({
                    'task': 'translation',
                    'id': f"trans_{len(task_data)+1:04d}",
                    'source_text': row['abstract_ko'].strip(),
                    'target_text': row['abstract_en'].strip(),
                    'source_lang': 'korean',
                    'target_lang': 'english',
                    'category': row.get('category', ''),
                    'metadata': {}
                })

            if len(task_data) >= target_size:
                break

            if pd.notna(row['content_ko']) and pd.notna(row['content_en']):
                task_data.append({
                    'task': 'translation',
                    'id': f"trans_{len(task_data)+1:04d}",
                    'source_text': row['content_ko'].strip(),
                    'target_text': row['content_en'].strip(),
                    'source_lang': 'korean',
                    'target_lang': 'english',
                    'category': row.get('category', ''),
                    'metadata': {}
                })

        # 4) 한문 → 영문 (간접)
        for idx, row in self.translated_df.iterrows():
            if len(task_data) >= target_size:
                break

            if pd.notna(row['abstract']) and pd.notna(row['abstract_en']):
                task_data.append({
                    'task': 'translation',
                    'id': f"trans_{len(task_data)+1:04d}",
                    'source_text': row['abstract'].strip(),
                    'target_text': row['abstract_en'].strip(),
                    'source_lang': 'classical_chinese',
                    'target_lang': 'english',
                    'category': row.get('category', ''),
                    'metadata': {'indirect': True}
                })

        print(f"  ✓ 생성 완료: {len(task_data)} 항목")
        print(f"    - 언어 쌍 분포:")
        lang_pairs = Counter([f"{item['source_lang']} → {item['target_lang']}" for item in task_data])
        for pair, count in lang_pairs.items():
            print(f"      {pair}: {count}")

        return task_data[:target_size]

    def generate_benchmark(self):
        """전체 벤치마크 생성"""
        print("\n" + "="*70)
        print("🚀 KLSBench 벤치마크 생성 시작")
        print("="*70)

        # 각 태스크 생성
        classification_data = self.generate_classification_task(2000)
        retrieval_data = self.generate_retrieval_task(2000)
        punctuation_data = self.generate_punctuation_task(2000)
        nli_sts_data = self.generate_nli_sts_task(2000)
        translation_data = self.generate_translation_task(2000)

        # 벤치마크 통합
        benchmark = {
            'benchmark_info': {
                'name': 'KLSBench',
                'full_name': 'Korean Classical Literature Understanding Benchmark',
                'version': '1.0',
                'description': 'C3Bench를 참고하여 개발한 한국 고전 문헌 이해 벤치마크',
                'tasks': ['classification', 'retrieval', 'punctuation', 'nli', 'translation'],
                'total_size': len(classification_data) + len(retrieval_data) +
                             len(punctuation_data) + len(nli_sts_data) + len(translation_data),
                'languages': ['Classical Chinese', 'Korean', 'English'],
                'data_sources': ['과거시험 데이터', '사서(四書) 데이터', 'NLI 예시']
            },
            'tasks': {
                'classification': {
                    'description': '주어진 고전 문헌의 문체(賦/詩/疑/義)를 분류',
                    'size': len(classification_data),
                    'metric': 'Accuracy',
                    'data': classification_data
                },
                'retrieval': {
                    'description': '주어진 문장이 유래한 원문의 출처(Book/Chapter)를 식별',
                    'size': len(retrieval_data),
                    'metric': 'Accuracy',
                    'data': retrieval_data
                },
                'punctuation': {
                    'description': '구두점이 없는 백문(白文)에 적절한 구두점을 복원',
                    'size': len(punctuation_data),
                    'metric': 'F1 Score',
                    'data': punctuation_data
                },
                'nli': {
                    'description': '두 문장 간의 논리적 관계(entailment/contradiction/neutral)를 판단',
                    'size': len(nli_sts_data),
                    'metric': 'Accuracy',
                    'data': nli_sts_data
                },
                'translation': {
                    'description': '한문, 한글, 영문 간의 번역 수행',
                    'size': len(translation_data),
                    'metric': 'BLEU Score',
                    'data': translation_data
                }
            }
        }

        # 저장
        self.save_benchmark(benchmark)

        # 통계 출력
        self.print_statistics(benchmark)

        return benchmark

    def save_benchmark(self, benchmark: Dict):
        """벤치마크 저장"""
        print("\n" + "="*70)
        print("💾 벤치마크 저장 중...")
        print("="*70)

        # 1) 전체 벤치마크 JSON 저장
        full_output_path = self.output_dir / 'kls_bench_full.json'
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 전체 벤치마크: {full_output_path}")

        # 2) 태스크별 개별 저장
        for task_name, task_info in benchmark['tasks'].items():
            task_output_path = self.output_dir / f'kls_bench_{task_name}.json'
            with open(task_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'task': task_name,
                    'description': task_info['description'],
                    'size': task_info['size'],
                    'metric': task_info['metric'],
                    'data': task_info['data']
                }, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {task_name}: {task_output_path}")

        # 3) CSV 형식으로도 저장 (분석 편의)
        for task_name, task_info in benchmark['tasks'].items():
            df = pd.DataFrame(task_info['data'])
            csv_output_path = self.output_dir / f'kls_bench_{task_name}.csv'
            df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ {task_name} (CSV): {csv_output_path}")

        # 4) README 생성
        self.generate_readme(benchmark)

    def generate_readme(self, benchmark: Dict):
        """README 문서 생성"""
        readme_path = self.output_dir / 'README.md'

        readme_content = f"""# KLSBench: Korean Classical Literature Understanding Benchmark

한국 고전 문헌 이해를 위한 포괄적인 벤치마크

## 📋 개요

**KLSBench**는 C3Bench를 참고하여 개발된 한국 고전 문헌 이해 벤치마크입니다.
대규모 언어 모델(LLM)의 한국 고전 한문 및 사서 데이터에 대한 이해 능력을 다각도로 평가합니다.

- **버전**: {benchmark['benchmark_info']['version']}
- **총 항목 수**: {benchmark['benchmark_info']['total_size']:,}개
- **태스크 수**: {len(benchmark['benchmark_info']['tasks'])}개
- **지원 언어**: {', '.join(benchmark['benchmark_info']['languages'])}

## 🎯 태스크 구성

| 태스크 | 설명 | 항목 수 | 평가 지표 |
|:---|:---|---:|:---|
"""

        for task_name, task_info in benchmark['tasks'].items():
            readme_content += f"| **{task_name}** | {task_info['description']} | {task_info['size']:,} | {task_info['metric']} |\n"

        readme_content += f"""
## 📊 데이터 통계

### 1. Classification (분류)

문체별 분포:
"""

        # Classification 통계
        cls_labels = [item['label'] for item in benchmark['tasks']['classification']['data']]
        cls_counts = Counter(cls_labels)
        for label, count in sorted(cls_counts.items()):
            readme_content += f"- **{label}**: {count:,}개\n"

        readme_content += """
### 2. Retrieval (검색)

책별 분포:
"""

        # Retrieval 통계
        ret_books = [item['book'] for item in benchmark['tasks']['retrieval']['data']]
        ret_counts = Counter(ret_books)
        for book, count in sorted(ret_counts.items(), key=lambda x: -x[1]):
            readme_content += f"- **{book}**: {count:,}개\n"

        readme_content += """
### 3. Punctuation (구두점)

평균 문장 길이 및 통계는 데이터 로딩 후 확인 가능합니다.

### 4. NLI (자연언어추론)

레이블 분포:
"""

        # NLI 통계
        nli_labels = [item['label'] for item in benchmark['tasks']['nli']['data']]
        nli_counts = Counter(nli_labels)
        for label, count in sorted(nli_counts.items()):
            readme_content += f"- **{label}**: {count:,}개\n"

        readme_content += """
### 5. Translation (번역)

언어 쌍 분포:
"""

        # Translation 통계
        trans_pairs = [f"{item['source_lang']} → {item['target_lang']}"
                       for item in benchmark['tasks']['translation']['data']]
        trans_counts = Counter(trans_pairs)
        for pair, count in sorted(trans_counts.items(), key=lambda x: -x[1]):
            readme_content += f"- **{pair}**: {count:,}개\n"

        readme_content += """
## 🚀 사용 방법

### Python에서 로드

```python
import json

# 전체 벤치마크 로드
with open('kls_bench_full.json', 'r', encoding='utf-8') as f:
    benchmark = json.load(f)

# 특정 태스크만 로드
with open('kls_bench_classification.json', 'r', encoding='utf-8') as f:
    classification_task = json.load(f)

# 데이터 접근
for item in classification_task['data']:
    print(f"Input: {item['input']}")
    print(f"Label: {item['label']}")
```

### Pandas로 분석

```python
import pandas as pd

# CSV로 로드
df = pd.read_csv('kls_bench_classification.csv')
print(df.head())
print(df['label'].value_counts())
```

## 📁 파일 구조

```
kls_bench/
├── kls_bench_full.json          # 전체 벤치마크 (모든 태스크 포함)
├── kls_bench_classification.json # 분류 태스크
├── kls_bench_retrieval.json     # 검색 태스크
├── kls_bench_punctuation.json   # 구두점 태스크
├── kls_bench_nli.json           # NLI 태스크
├── kls_bench_translation.json   # 번역 태스크
├── kls_bench_classification.csv # 분류 태스크 (CSV)
├── kls_bench_retrieval.csv      # 검색 태스크 (CSV)
├── kls_bench_punctuation.csv    # 구두점 태스크 (CSV)
├── kls_bench_nli.csv            # NLI 태스크 (CSV)
├── kls_bench_translation.csv    # 번역 태스크 (CSV)
└── README.md                          # 본 문서
```

## 🎓 데이터 출처

1. **과거시험 데이터**: 한국 과거시험 문제 및 답안 (문체 분류 포함)
2. **사서(四書) 데이터**: 논어, 맹자, 대학, 중용 등 유교 경전
3. **NLI 예시**: 자연언어추론 템플릿 및 예시

## 📜 라이선스 및 인용

이 벤치마크를 연구에 사용하시는 경우 다음과 같이 인용해 주세요:

```bibtex
@misc{{kls_bench_2024,
  title={{KLSBench: Korean Classical Literature Understanding Benchmark}},
  author={{Your Name}},
  year={{2024}},
  note={{Inspired by C3Bench}}
}}
```

## 🔗 참고 자료

- **C3Bench**: [논문 링크]
- **관련 연구**: 고전 한문 자연어처리 연구

## 📧 문의

벤치마크 관련 문의사항은 이메일로 연락 주시기 바랍니다.

---

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"  ✓ README: {readme_path}")

    def print_statistics(self, benchmark: Dict):
        """벤치마크 통계 출력"""
        print("\n" + "="*70)
        print("📊 벤치마크 통계")
        print("="*70)

        print(f"\n🎯 전체 요약")
        print(f"  - 벤치마크 이름: {benchmark['benchmark_info']['name']}")
        print(f"  - 총 항목 수: {benchmark['benchmark_info']['total_size']:,}개")
        print(f"  - 태스크 수: {len(benchmark['benchmark_info']['tasks'])}개")

        print(f"\n📋 태스크별 통계:")
        for task_name, task_info in benchmark['tasks'].items():
            print(f"\n  [{task_name.upper()}]")
            print(f"    - 항목 수: {task_info['size']:,}개")
            print(f"    - 평가 지표: {task_info['metric']}")

            # 추가 통계
            if task_name == 'classification':
                labels = [item['label'] for item in task_info['data']]
                label_counts = Counter(labels)
                print(f"    - 레이블 분포: {dict(label_counts)}")

            elif task_name == 'nli':
                labels = [item['label'] for item in task_info['data']]
                label_counts = Counter(labels)
                print(f"    - 레이블 분포: {dict(label_counts)}")

            elif task_name == 'translation':
                pairs = [f"{item['source_lang']}→{item['target_lang']}"
                        for item in task_info['data']]
                pair_counts = Counter(pairs)
                print(f"    - 언어 쌍 분포:")
                for pair, count in pair_counts.most_common():
                    print(f"      • {pair}: {count:,}개")

        print("\n" + "="*70)
        print("✅ 벤치마크 생성 완료!")
        print("="*70)


def main():
    """메인 실행 함수"""
    # 경로 설정
    translated_csv_path = "/Users/songhune/Workspace/korean_eda/notebook/experiments/graphs/translated_full_20251020_212605.csv"
    external_csv_path = "/Users/songhune/Workspace/korean_eda/data/External_raw.csv"
    nli_examples_path = "/Users/songhune/Workspace/korean_eda/examples/nli_examples.json"
    output_dir = "/Users/songhune/Workspace/korean_eda/benchmark/kls_bench"

    # 벤치마크 생성기 초기화
    generator = KLSBenchGenerator(
        translated_csv_path=translated_csv_path,
        external_csv_path=external_csv_path,
        nli_examples_path=nli_examples_path,
        output_dir=output_dir
    )

    # 벤치마크 생성
    benchmark = generator.generate_benchmark()

    print(f"\n🎉 벤치마크가 성공적으로 생성되었습니다!")
    print(f"📁 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
