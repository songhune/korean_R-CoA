#!/usr/bin/env python3
"""
Configuration Loader for KLSBench Evaluation
Provides utilities to read and access config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class Config:
    """Configuration loader and accessor for KLSBench evaluation"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            config_path: Path to config.yaml (default: auto-detect)
        """
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.yaml"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    # Benchmark paths
    def get_benchmark_path(self, task: Optional[str] = None) -> str:
        """
        Get benchmark file path

        Args:
            task: Task name (classification, nli, etc.) or None for full benchmark

        Returns:
            Absolute or relative path to benchmark JSON file
        """
        if task is None or task == 'full':
            return self.config['benchmark']['full']
        return self.config['benchmark'].get(task)

    def get_output_dir(self, output_type: str = 'base') -> str:
        """
        Get output directory path

        Args:
            output_type: 'base', 'fewshot', or 'aggregated'

        Returns:
            Path to output directory
        """
        return self.config['output'][output_type]

    # Model configurations
    def get_api_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get API model configurations

        Args:
            provider: 'openai' or 'anthropic' (None for all)

        Returns:
            List of model configurations
        """
        api_models = self.config['models']['api']

        if provider is None:
            # Return all API models
            all_models = []
            for p in api_models.values():
                all_models.extend(p)
            return all_models

        return api_models.get(provider, [])

    def get_opensource_models(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get open-source model configurations

        Args:
            enabled_only: Return only enabled models

        Returns:
            List of model configurations
        """
        models = self.config['models']['opensource']
        if enabled_only:
            return [m for m in models if m.get('enabled', True)]
        return models

    def get_supervised_models(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get supervised learning model configurations

        Args:
            enabled_only: Return only enabled models

        Returns:
            List of model configurations
        """
        models = self.config['models']['supervised']
        if enabled_only:
            return [m for m in models if m.get('enabled', True)]
        return models

    def get_all_models(self, enabled_only: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all model configurations organized by type

        Args:
            enabled_only: Return only enabled models

        Returns:
            Dictionary with keys: 'api', 'opensource', 'supervised'
        """
        return {
            'api': self.get_api_models(),
            'opensource': self.get_opensource_models(enabled_only),
            'supervised': self.get_supervised_models(enabled_only)
        }

    # Evaluation modes
    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """
        Get evaluation mode configuration

        Args:
            mode: 'test', 'sample', or 'full'

        Returns:
            Mode configuration dictionary
        """
        return self.config['modes'].get(mode, {})

    def get_fewshot_config(self) -> Dict[str, Any]:
        """Get few-shot evaluation configuration"""
        return self.config['fewshot']

    # Task information
    def get_task_info(self, task: Optional[str] = None) -> Dict[str, Any]:
        """
        Get task information

        Args:
            task: Task name or None for all tasks

        Returns:
            Task information dictionary
        """
        if task is None:
            return self.config['tasks']
        return self.config['tasks'].get(task, {})

    def get_total_items(self, task: Optional[str] = None) -> int:
        """
        Get total number of items for a task

        Args:
            task: Task name or None for total across all tasks

        Returns:
            Total item count
        """
        if task is None:
            return sum(t['total_items'] for t in self.config['tasks'].values())
        return self.config['tasks'].get(task, {}).get('total_items', 0)

    def calculate_sample_size(self, ratio: float, task: Optional[str] = None) -> int:
        """
        Calculate sample size based on ratio

        Args:
            ratio: Sampling ratio (0.0 to 1.0)
            task: Task name or None for total

        Returns:
            Number of samples
        """
        total = self.get_total_items(task)
        return max(1, int(total * ratio))

    # Utility methods
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("KLSBench Configuration Summary")
        print("=" * 60)
        print(f"\nBenchmark Path: {self.get_benchmark_path()}")
        print(f"Output Directory: {self.get_output_dir()}")

        print("\nTasks:")
        for task_name, task_info in self.config['tasks'].items():
            print(f"  - {task_name}: {task_info['total_items']} items ({task_info['metric']})")

        print(f"\nTotal Items: {self.get_total_items()}")

        print("\nAPI Models:")
        for provider, models in self.config['models']['api'].items():
            print(f"  {provider}:")
            for model in models:
                status = "enabled" if model.get('enabled', True) else "disabled"
                print(f"    - {model['name']} ({status})")

        print("\nOpen Source Models:")
        for model in self.get_opensource_models(enabled_only=False):
            status = "enabled" if model.get('enabled', True) else "disabled"
            print(f"  - {model['name']} ({status})")

        print("\nSupervised Models:")
        for model in self.get_supervised_models(enabled_only=False):
            status = "enabled" if model.get('enabled', True) else "disabled"
            note = f" - {model['note']}" if 'note' in model else ""
            print(f"  - {model['name']} ({status}){note}")

        print("\nFew-shot Configuration:")
        fewshot = self.get_fewshot_config()
        print(f"  - Enabled: {fewshot['enabled']}")
        print(f"  - Shots: {fewshot['shots']}")
        print(f"  - Max samples: {fewshot['max_samples']}")
        print(f"  - Tasks: {', '.join(fewshot['tasks'])}")

        print("=" * 60)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="KLSBench Configuration Utility")
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--summary', action='store_true', help='Print configuration summary')
    parser.add_argument('--benchmark', type=str, help='Get benchmark path for task')
    parser.add_argument('--models', choices=['api', 'opensource', 'supervised', 'all'],
                       help='List models')
    parser.add_argument('--sample-size', type=float, help='Calculate sample size for ratio')
    parser.add_argument('--task', type=str, help='Specific task name')

    args = parser.parse_args()

    config = Config(args.config)

    if args.summary:
        config.print_summary()

    if args.benchmark:
        path = config.get_benchmark_path(args.benchmark)
        print(f"Benchmark path: {path}")

    if args.models:
        if args.models == 'api':
            models = config.get_api_models()
            print("API Models:")
            for m in models:
                print(f"  - {m['name']}")
        elif args.models == 'opensource':
            models = config.get_opensource_models()
            print("Open Source Models:")
            for m in models:
                print(f"  - {m['name']}")
        elif args.models == 'supervised':
            models = config.get_supervised_models()
            print("Supervised Models:")
            for m in models:
                print(f"  - {m['name']}")
        elif args.models == 'all':
            all_models = config.get_all_models()
            for model_type, models in all_models.items():
                print(f"\n{model_type.upper()} Models:")
                for m in models:
                    print(f"  - {m['name']}")

    if args.sample_size is not None:
        size = config.calculate_sample_size(args.sample_size, args.task)
        task_str = f" for {args.task}" if args.task else ""
        print(f"Sample size at {args.sample_size*100:.0f}%{task_str}: {size} items")


if __name__ == '__main__':
    main()
