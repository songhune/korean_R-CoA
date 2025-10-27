#!/usr/bin/env python3
"""
Experiment 6: Few-shot Evaluation
=====================================

Zero-shot에서 0% 성능을 보인 모델들에 대해 few-shot (1-shot, 3-shot) 평가 수행.
특히 Classification과 NLI 태스크에 집중.

Target models:
- Claude 3.5 Sonnet (NLI 0%)
- Llama 3.1 8B (Classification 20%, NLI 0%)
- Qwen 2.5 7B (Classification 0%, NLI 0%)

Target tasks:
- Classification (대부분 0%)
- NLI (4개 모델 0%)

Expected timeline: 1-2 days
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))


class FewShotEvaluator:
    """Few-shot evaluation for JC2Bench"""

    def __init__(self, benchmark_path: str, results_dir: str = "../../results/fewshot"):
        """
        Initialize few-shot evaluator

        Args:
            benchmark_path: Path to benchmark JSON
            results_dir: Directory to save results
        """
        self.benchmark_path = benchmark_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load benchmark
        print(f"Loading benchmark from {benchmark_path}...")
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)

        print(f"Benchmark loaded: {self.benchmark['benchmark_info']['total_size']} total instances")

    def select_exemplars(
        self,
        task_data: List[Dict],
        n_shots: int,
        seed: int = 42
    ) -> List[Dict]:
        """
        Select exemplars for few-shot learning

        Args:
            task_data: Task data
            n_shots: Number of shots
            seed: Random seed

        Returns:
            List of exemplars
        """
        random.seed(seed)

        # For classification/NLI, balance across labels
        if 'label' in task_data[0]:
            # Get unique labels
            label_to_data = {}
            for item in task_data:
                label = item['label']
                if label not in label_to_data:
                    label_to_data[label] = []
                label_to_data[label].append(item)

            # Sample equally from each label
            exemplars = []
            labels = list(label_to_data.keys())
            per_label = max(1, n_shots // len(labels))

            for label in labels:
                samples = random.sample(label_to_data[label], min(per_label, len(label_to_data[label])))
                exemplars.extend(samples)

            # If we need more, add random samples
            while len(exemplars) < n_shots:
                exemplars.append(random.choice(task_data))

            return exemplars[:n_shots]
        else:
            # Random sampling for other tasks
            return random.sample(task_data, n_shots)

    def format_classification_fewshot(
        self,
        exemplars: List[Dict],
        test_item: Dict
    ) -> Tuple[str, str]:
        """Format classification task with few-shot examples"""

        system_prompt = """You are an expert in classical Chinese literature and Korean historical texts.
Your task is to classify the literary style (文體, munche) of given texts from Joseon Dynasty civil service examinations.

Here are some examples:"""

        # Add exemplars
        for i, ex in enumerate(exemplars, 1):
            system_prompt += f"\n\nExample {i}:"
            system_prompt += f"\nText: {ex['input']}"
            system_prompt += f"\nLiterary Style: {ex['label']}"

        user_prompt = f"""Now classify the following text:

Text: {test_item['input']}

Respond with ONLY the literary style name (賦, 詩, 疑, 義, 策, 表, 論, 銘, 箋, etc.)."""

        return system_prompt, user_prompt

    def format_nli_fewshot(
        self,
        exemplars: List[Dict],
        test_item: Dict
    ) -> Tuple[str, str]:
        """Format NLI task with few-shot examples"""

        system_prompt = """You are an expert in classical Chinese philosophy and Korean classical literature.
Your task is to determine the logical relationship between two statements from classical texts.

The relationship can be:
- entailment: The hypothesis logically follows from the premise
- contradiction: The hypothesis contradicts the premise
- neutral: No logical relationship exists

Here are some examples:"""

        # Add exemplars
        for i, ex in enumerate(exemplars, 1):
            system_prompt += f"\n\nExample {i}:"
            system_prompt += f"\nPremise: {ex['premise']}"
            system_prompt += f"\nHypothesis: {ex['hypothesis']}"
            system_prompt += f"\nRelationship: {ex['label']}"

        user_prompt = f"""Now determine the relationship for:

Premise: {test_item['premise']}
Hypothesis: {test_item['hypothesis']}

Respond with ONLY one word: entailment, contradiction, or neutral."""

        return system_prompt, user_prompt

    def evaluate_model_fewshot(
        self,
        model_wrapper,
        task_name: str,
        n_shots: int,
        max_samples: int = 50  # Reduced for faster evaluation
    ) -> Dict:
        """
        Evaluate model with few-shot examples

        Args:
            model_wrapper: Model wrapper with generate() method
            task_name: Task name (classification or nli)
            n_shots: Number of shots (1, 3, 5)
            max_samples: Maximum samples to evaluate

        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {task_name} with {n_shots}-shot")
        print(f"{'='*60}")

        # Get task data
        task_data = self.benchmark['tasks'][task_name]['data']

        # Limit samples for faster evaluation
        if max_samples and max_samples < len(task_data):
            task_data = random.sample(task_data, max_samples)

        # Select exemplars (from beginning, not overlapping with test)
        all_data = self.benchmark['tasks'][task_name]['data']
        exemplars = self.select_exemplars(all_data[:100], n_shots)

        # Ensure test data doesn't overlap with exemplars
        exemplar_ids = {ex.get('id', ex.get('question_id')) for ex in exemplars}
        test_data = [item for item in task_data
                     if item.get('id', item.get('question_id')) not in exemplar_ids]

        # Evaluate
        predictions = []
        ground_truths = []

        for item in tqdm(test_data, desc=f"{n_shots}-shot {task_name}"):
            try:
                # Format prompt based on task
                if task_name == 'classification':
                    system_prompt, user_prompt = self.format_classification_fewshot(
                        exemplars, item
                    )
                    ground_truth = item['label']
                elif task_name == 'nli':
                    system_prompt, user_prompt = self.format_nli_fewshot(
                        exemplars, item
                    )
                    ground_truth = item['label']
                else:
                    raise ValueError(f"Unsupported task: {task_name}")

                # Generate prediction
                prediction = model_wrapper.generate(system_prompt, user_prompt)
                prediction = prediction.strip()

                predictions.append(prediction)
                ground_truths.append(ground_truth)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error on item: {e}")
                predictions.append("")
                ground_truths.append(ground_truth if 'ground_truth' in locals() else "")

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        accuracy = accuracy_score(ground_truths, predictions)
        try:
            f1 = f1_score(ground_truths, predictions, average='macro', zero_division=0)
        except:
            f1 = 0.0

        print(f"\n{n_shots}-shot Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        try:
            print("\nClassification Report:")
            print(classification_report(ground_truths, predictions, zero_division=0))
        except:
            pass

        return {
            'task': task_name,
            'n_shots': n_shots,
            'num_samples': len(test_data),
            'accuracy': accuracy,
            'f1': f1,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'exemplars': exemplars
        }

    def run_fewshot_experiment(
        self,
        model_name: str,
        model_type: str,
        api_key: Optional[str] = None,
        shots: List[int] = [1, 3],
        tasks: List[str] = ['classification', 'nli'],
        max_samples: int = 50
    ):
        """
        Run few-shot experiment for a model

        Args:
            model_name: Model name
            model_type: 'api' or 'opensource'
            api_key: API key for API models
            shots: List of shot numbers to evaluate
            tasks: List of tasks to evaluate
            max_samples: Max samples per task
        """
        print(f"\n{'#'*60}")
        print(f"Few-shot Experiment: {model_name}")
        print(f"{'#'*60}")

        # Import model wrapper
        from exp5_benchmark_evaluation import (
            OpenAIWrapper, AnthropicWrapper, HuggingFaceWrapper
        )

        # Initialize model
        if model_type == 'api':
            if 'gpt' in model_name.lower():
                model_wrapper = OpenAIWrapper(model_name, api_key)
            elif 'claude' in model_name.lower():
                model_wrapper = AnthropicWrapper(model_name, api_key)
            else:
                raise ValueError(f"Unknown API model: {model_name}")
        elif model_type == 'opensource':
            model_wrapper = HuggingFaceWrapper(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Run experiments
        all_results = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'shots': shots,
            'tasks': {}
        }

        for task in tasks:
            all_results['tasks'][task] = {}

            for n_shot in shots:
                result = self.evaluate_model_fewshot(
                    model_wrapper, task, n_shot, max_samples
                )
                all_results['tasks'][task][f'{n_shot}-shot'] = result

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_')
        output_path = self.results_dir / f"fewshot_{safe_model_name}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

        # Save summary
        summary_data = []
        for task in tasks:
            for n_shot in shots:
                result = all_results['tasks'][task][f'{n_shot}-shot']
                summary_data.append({
                    'model': model_name,
                    'task': task,
                    'n_shots': n_shot,
                    'accuracy': result['accuracy'],
                    'f1': result['f1'],
                    'num_samples': result['num_samples']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / f"summary_{safe_model_name}_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"Summary saved to: {summary_path}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))

        return all_results


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Few-shot evaluation for JC2Bench')
    parser.add_argument('--benchmark', type=str,
                       default='../../benchmark/k_classic_bench/k_classic_bench_full.json',
                       help='Path to benchmark JSON')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--model-type', type=str, choices=['api', 'opensource'],
                       default='api', help='Model type')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (for API models)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 3],
                       help='Number of shots to evaluate (e.g., 1 3 5)')
    parser.add_argument('--tasks', type=str, nargs='+',
                       default=['classification', 'nli'],
                       help='Tasks to evaluate')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Max samples per task (for faster evaluation)')
    parser.add_argument('--output', type=str, default='../../results/fewshot',
                       help='Output directory')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = FewShotEvaluator(args.benchmark, args.output)

    # Run experiment
    evaluator.run_fewshot_experiment(
        model_name=args.model_name,
        model_type=args.model_type,
        api_key=args.api_key,
        shots=args.shots,
        tasks=args.tasks,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
