"""
R-CoA Data Preprocessing Pipeline
Cross-lingual pairs 생성 (Classical Chinese ↔ Modern Chinese ↔ Korean ↔ English)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class CrossLingualPair:
    """Cross-lingual pair for InfoNCE training"""
    classical_chinese: str
    modern_chinese: str
    korean: str = None
    english: str = None
    task: str = ""

    def to_dict(self):
        return {
            'classical_chinese': self.classical_chinese,
            'modern_chinese': self.modern_chinese,
            'korean': self.korean,
            'english': self.english,
            'task': self.task
        }


class RCoADataPreprocessor:
    """
    R-CoA 데이터 전처리

    Sources:
    1. ACCN-INS.json: TongGu model's Classical↔Modern Chinese pairs
    2. combined_ACCN-INS_chunks.jsonl: Korean/English translations (partial)
    3. KLSBench: Korean classical literature benchmark
    """

    def __init__(self,
                 accn_path: str = "/home/work/songhune/ACCN-INS.json",
                 combined_path: str = "/home/work/songhune/korean_R-CoA/experiments/tongu-translate/combined_ACCN-INS_chunks.jsonl",
                 klsbench_path: str = "/home/work/songhune/korean_R-CoA/benchmark/kls_bench_full.json"):
        self.accn_path = Path(accn_path)
        self.combined_path = Path(combined_path)
        self.klsbench_path = Path(klsbench_path)

        self.data = []

    def load_accn_ins(self) -> List[Dict]:
        """Load ACCN-INS dataset (Classical ↔ Modern Chinese)"""
        print(f"[LOAD] Loading ACCN-INS from {self.accn_path}")
        with open(self.accn_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"  Total samples: {len(data):,}")
        return data

    def load_combined_translations(self) -> List[Dict]:
        """Load combined ACCN-INS with Korean/English translations"""
        print(f"[LOAD] Loading translations from {self.combined_path}")
        data = []
        with open(self.combined_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        print(f"  Total translated samples: {len(data):,}")
        return data

    def extract_classical_chinese_from_accn(self, accn_item: Dict) -> str:
        """Extract classical Chinese text from ACCN item"""
        data_dict = accn_item.get('data', {})

        # Try to find classical Chinese in history or instruction
        if 'history' in data_dict and data_dict['history']:
            # First element of first history entry usually contains classical text
            return data_dict['history'][0][0]

        # Fallback: extract from instruction if it contains classical Chinese
        instruction = data_dict.get('instruction', '')
        if '翻译' in instruction or '古文' in instruction:
            # Try to extract text after colon or question mark
            for delimiter in ['：', ':', '：', '。', '，']:
                if delimiter in instruction:
                    parts = instruction.split(delimiter)
                    if len(parts) > 1:
                        return parts[-1].strip()

        return None

    def create_cross_lingual_pairs(self) -> List[CrossLingualPair]:
        """Create cross-lingual pairs from all sources"""
        print("\n[CREATE] Building cross-lingual pairs...")

        # Load datasets
        accn_data = self.load_accn_ins()
        translated_data = self.load_combined_translations()

        pairs = []

        # 1. Process translated data (has all 4 languages)
        print(f"\n[PROCESS] Processing translated data...")
        for item in translated_data:
            classical = self.extract_classical_chinese_from_accn(item)
            modern = item['data'].get('output', '')
            korean = item.get('korean_translation', '')
            english = item.get('english_translation', '')
            task = item.get('task', '')

            if classical and modern and (korean or english):
                pair = CrossLingualPair(
                    classical_chinese=classical,
                    modern_chinese=modern,
                    korean=korean if korean else None,
                    english=english if english else None,
                    task=task
                )
                pairs.append(pair)

        print(f"  Created {len(pairs):,} pairs with translations")

        # 2. Process remaining ACCN data (Classical ↔ Modern Chinese only)
        print(f"\n[PROCESS] Processing ACCN data (Chinese only)...")
        translated_classicals = {self.extract_classical_chinese_from_accn(item)
                                for item in translated_data}

        chinese_only_pairs = 0
        for item in accn_data:
            classical = self.extract_classical_chinese_from_accn(item)
            modern = item['data'].get('output', '')
            task = item.get('task', '')

            # Skip if already in translated data
            if classical in translated_classicals:
                continue

            if classical and modern and len(classical) > 5:
                pair = CrossLingualPair(
                    classical_chinese=classical,
                    modern_chinese=modern,
                    korean=None,
                    english=None,
                    task=task
                )
                pairs.append(pair)
                chinese_only_pairs += 1

                # Limit to avoid too much data
                if chinese_only_pairs >= 10000:
                    break

        print(f"  Added {chinese_only_pairs:,} Chinese-only pairs")
        print(f"\n[SUMMARY] Total pairs: {len(pairs):,}")

        return pairs

    def save_pairs(self, pairs: List[CrossLingualPair], output_path: str):
        """Save cross-lingual pairs to JSONL"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] Saving to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')

        print(f"  Saved {len(pairs):,} pairs")

    def create_train_val_split(self, pairs: List[CrossLingualPair],
                               val_ratio: float = 0.1) -> Tuple[List, List]:
        """Split data into train/validation sets"""
        random.seed(42)
        random.shuffle(pairs)

        val_size = int(len(pairs) * val_ratio)
        val_pairs = pairs[:val_size]
        train_pairs = pairs[val_size:]

        print(f"\n[SPLIT] Train/Val split:")
        print(f"  Train: {len(train_pairs):,} pairs")
        print(f"  Val:   {len(val_pairs):,} pairs")

        return train_pairs, val_pairs

    def generate_statistics(self, pairs: List[CrossLingualPair]):
        """Generate dataset statistics"""
        stats = {
            'total': len(pairs),
            'with_korean': sum(1 for p in pairs if p.korean),
            'with_english': sum(1 for p in pairs if p.english),
            'with_both_translations': sum(1 for p in pairs if p.korean and p.english),
            'chinese_only': sum(1 for p in pairs if not p.korean and not p.english)
        }

        # Length statistics
        classical_lengths = [len(p.classical_chinese) for p in pairs]
        modern_lengths = [len(p.modern_chinese) for p in pairs]

        stats['avg_classical_length'] = sum(classical_lengths) / len(classical_lengths)
        stats['avg_modern_length'] = sum(modern_lengths) / len(modern_lengths)

        print("\n[STATISTICS]")
        print(f"  Total pairs: {stats['total']:,}")
        print(f"  With Korean: {stats['with_korean']:,} ({stats['with_korean']/stats['total']*100:.1f}%)")
        print(f"  With English: {stats['with_english']:,} ({stats['with_english']/stats['total']*100:.1f}%)")
        print(f"  With both translations: {stats['with_both_translations']:,}")
        print(f"  Chinese only: {stats['chinese_only']:,}")
        print(f"  Avg classical length: {stats['avg_classical_length']:.1f} chars")
        print(f"  Avg modern length: {stats['avg_modern_length']:.1f} chars")

        return stats


def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("R-CoA Data Preprocessing Pipeline")
    print("="*70)

    # Initialize preprocessor
    preprocessor = RCoADataPreprocessor()

    # Create cross-lingual pairs
    pairs = preprocessor.create_cross_lingual_pairs()

    # Generate statistics
    stats = preprocessor.generate_statistics(pairs)

    # Train/Val split
    train_pairs, val_pairs = preprocessor.create_train_val_split(pairs, val_ratio=0.1)

    # Save datasets
    output_dir = Path("/home/work/songhune/korean_R-CoA/experiments/rcoa/data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_pairs(train_pairs, output_dir / "train_pairs.jsonl")
    preprocessor.save_pairs(val_pairs, output_dir / "val_pairs.jsonl")

    # Save statistics
    with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("[COMPLETE] Data preprocessing finished")
    print("="*70)


if __name__ == "__main__":
    main()
