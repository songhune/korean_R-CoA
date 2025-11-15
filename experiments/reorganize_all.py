#!/usr/bin/env python3
"""
Reorganize all experiment outputs to consolidated results directory
Updates all script paths and cleans up duplicate files
"""

import re
import shutil
from pathlib import Path
from typing import List, Tuple


class Reorganizer:
    """Reorganize and clean up results structure"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir.parent.parent / 'results'

    def update_python_files(self):
        """Update all Python script paths"""
        print("\n=== Updating Python Scripts ===")

        # Pattern mappings (old_pattern -> new_pattern)
        replacements = [
            # Old server paths
            (r'../../results/raw_evaluation', '../../results/raw_evaluation'),
            (r'../../results/raw_evaluation/aggregated', '../../results/aggregated'),

            # Old benchmark paths
            (r'../../results/raw_evaluation', '../../results/raw_evaluation'),
            (r'../../results/raw_evaluation/aggregated', '../../results/aggregated'),

            # Graphs path
            (r'../../results/figures', '../../results/figures'),
            (r"output_dir = Path\(__file__\)\.parent\.parent / 'graphs'",
             "output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'"),

            # Default arguments
            (r"default='../../results/raw_evaluation'",
             "default='../../results/raw_evaluation'"),
            (r"default='../../results/raw_evaluation/aggregated'",
             "default='../../results/aggregated'"),
            (r"default='../../results/figures'", "default='../../results/figures'"),
        ]

        # Find all Python files
        py_files = list(self.base_dir.rglob("*.py"))

        for py_file in py_files:
            if '__pycache__' in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content

                for old_pattern, new_pattern in replacements:
                    content = re.sub(old_pattern, new_pattern, content)

                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    print(f"  ✓ Updated: {py_file.relative_to(self.base_dir)}")

            except Exception as e:
                print(f"  ⚠ Failed to update {py_file}: {e}")

    def update_shell_files(self):
        """Update all shell script paths"""
        print("\n=== Updating Shell Scripts ===")

        replacements = [
            (r'OUTPUT_DIR="../../results/figures"', 'OUTPUT_DIR="../../results/figures"'),
            (r'--output-dir ../../results/figures', '--output-dir ../../results/figures'),
            (r'--results-dir ../../results/raw_evaluation', '--results-dir ../../results/raw_evaluation'),
        ]

        sh_files = list(self.base_dir.rglob("*.sh"))

        for sh_file in sh_files:
            try:
                content = sh_file.read_text(encoding='utf-8')
                original_content = content

                for old_pattern, new_pattern in replacements:
                    content = re.sub(old_pattern, new_pattern, content)

                if content != original_content:
                    sh_file.write_text(content, encoding='utf-8')
                    print(f"  ✓ Updated: {sh_file.relative_to(self.base_dir)}")

            except Exception as e:
                print(f"  ⚠ Failed to update {sh_file}: {e}")

    def clean_old_directories(self):
        """Clean up old files from graphs directory"""
        print("\n=== Cleaning Up Old Directories ===")

        graphs_dir = self.base_dir / 'graphs'

        if graphs_dir.exists():
            # Check what's left in graphs
            remaining = list(graphs_dir.iterdir())
            print(f"  Remaining files in graphs: {len(remaining)}")

            # If only hidden files or backups left, can be safely removed
            can_remove = all(f.name.startswith('.') or f.name.endswith('.bak')
                           for f in remaining)

            if can_remove and remaining:
                print(f"  Cleaning up {len(remaining)} temporary files")
                for f in remaining:
                    if f.is_file():
                        f.unlink()

    def verify_structure(self):
        """Verify the new directory structure"""
        print("\n=== Verifying Directory Structure ===")

        expected_dirs = [
            self.results_dir / 'raw_evaluation',
            self.results_dir / 'aggregated',
            self.results_dir / 'figures' / 'appendix_a',
            self.results_dir / 'figures' / 'appendix_b',
            self.results_dir / 'figures' / 'detailed',
            self.results_dir / 'figures' / 'radar',
            self.results_dir / 'figures' / 'legacy',
            self.results_dir / 'tables' / 'examples',
            self.results_dir / 'tables' / 'statistics',
            self.results_dir / 'tables' / 'performance',
            self.results_dir / 'data_processing',
        ]

        all_exist = True
        for d in expected_dirs:
            exists = d.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {d.relative_to(self.results_dir.parent)}")
            if not exists:
                all_exist = False

        return all_exist

    def count_files(self):
        """Count files in each directory"""
        print("\n=== File Count Summary ===")

        dirs_to_check = {
            'Raw Evaluation': self.results_dir / 'raw_evaluation',
            'Aggregated': self.results_dir / 'aggregated',
            'Figures - Appendix A': self.results_dir / 'figures' / 'appendix_a',
            'Figures - Appendix B': self.results_dir / 'figures' / 'appendix_b',
            'Figures - Detailed': self.results_dir / 'figures' / 'detailed',
            'Figures - Radar': self.results_dir / 'figures' / 'radar',
            'Figures - Legacy': self.results_dir / 'figures' / 'legacy',
            'Tables - Examples': self.results_dir / 'tables' / 'examples',
            'Tables - Statistics': self.results_dir / 'tables' / 'statistics',
            'Tables - Performance': self.results_dir / 'tables' / 'performance',
            'Data Processing': self.results_dir / 'data_processing',
        }

        total = 0
        for name, d in dirs_to_check.items():
            if d.exists():
                count = len(list(d.iterdir()))
                total += count
                print(f"  {name:30s}: {count:3d} files")
            else:
                print(f"  {name:30s}: (not found)")

        print(f"\n  {'Total':30s}: {total:3d} files")

    def run_all(self):
        """Run all reorganization steps"""
        print("="*70)
        print("Results Directory Reorganization")
        print("="*70)

        self.update_python_files()
        self.update_shell_files()
        self.clean_old_directories()

        print("\n" + "="*70)
        print("Verification")
        print("="*70)

        structure_ok = self.verify_structure()
        self.count_files()

        print("\n" + "="*70)
        if structure_ok:
            print("✅ Reorganization Complete!")
        else:
            print("⚠ Some directories are missing")
        print("="*70)


def main():
    base_dir = Path(__file__).parent
    reorganizer = Reorganizer(str(base_dir))
    reorganizer.run_all()


if __name__ == '__main__':
    main()
