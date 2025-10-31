#!/usr/bin/env python3
"""
Generate Radar/Spider Charts for Model Performance
Creates publication-quality radar charts showing model performance across all 5 tasks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns

from font_fix import setup_korean_fonts_robust


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                            path=MplPath.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


class RadarChartGenerator:
    """Generate various radar chart visualizations"""

    def __init__(self, results_csv: str, output_dir: str):
        self.results_csv = Path(results_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup fonts
        korean_font = setup_korean_fonts_robust()
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', korean_font] if korean_font else ['DejaVu Sans']

        # Load data
        self.df = pd.read_csv(results_csv, encoding='utf-8-sig')
        self.pivot_df = self._create_pivot_table()

    def _create_pivot_table(self):
        """Create model x task pivot table"""
        pivot = self.df.pivot_table(
            index='model',
            columns='task',
            values='primary_metric',
            aggfunc='first'
        )
        # Ensure task order
        task_order = ['classification', 'retrieval', 'punctuation', 'nli', 'translation']
        pivot = pivot[task_order]
        return pivot

    def generate_all_radar_charts(self):
        """Generate all radar chart variations"""
        print("\n" + "="*70)
        print("Generating Radar Charts")
        print("="*70)

        # 1. Individual model radar charts (all models overlaid)
        self.create_all_models_radar()

        # 2. Model type comparison (API vs Opensource)
        self.create_model_type_comparison()

        # 3. Top models comparison
        self.create_top_models_comparison()

        # 4. Individual model cards (separate charts)
        self.create_individual_model_cards()

        # 5. Grid of small multiples
        self.create_small_multiples_grid()

        print("\n✅ All radar charts generated!")
        print(f"📁 Output directory: {self.output_dir}")

    def create_all_models_radar(self):
        """Create radar chart with all models overlaid"""
        print("\n📊 Creating all-models radar chart...")

        # Prepare data
        tasks = ['Classification', 'Retrieval', 'Punctuation', 'NLI', 'Translation']
        num_vars = len(tasks)

        theta = radar_factory(num_vars, frame='polygon')

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.pivot_df)))

        # Plot each model
        for idx, (model_name, row) in enumerate(self.pivot_df.iterrows()):
            values = row.values
            values = np.concatenate((values, [values[0]]))  # Close the polygon

            # Shorten model name
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name

            ax.plot(theta, values[:-1], 'o-', linewidth=2, label=short_name,
                   color=colors[idx], markersize=6)
            ax.fill(theta, values[:-1], alpha=0.15, color=colors[idx])

        ax.set_varlabels(tasks)
        ax.set_ylim(0, 1.0)
        ax.set_title('KLSBench: Model Performance Across All Tasks',
                    position=(0.5, 1.1), ha='center', fontsize=16, fontweight='bold')

        # Legend outside the plot
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'radar_all_models.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def create_model_type_comparison(self):
        """Compare API models vs Open-source models"""
        print("\n📊 Creating model type comparison radar chart...")

        # Add model type to pivot
        model_types = self.df.groupby('model')['model_type'].first()

        # Calculate average by model type
        api_models = [m for m in self.pivot_df.index if model_types[m] == 'api']
        opensource_models = [m for m in self.pivot_df.index if model_types[m] == 'opensource']

        api_avg = self.pivot_df.loc[api_models].mean()
        opensource_avg = self.pivot_df.loc[opensource_models].mean()

        # Prepare data
        tasks = ['Classification', 'Retrieval', 'Punctuation', 'NLI', 'Translation']
        num_vars = len(tasks)

        theta = radar_factory(num_vars, frame='polygon')

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

        # API models (average)
        values_api = api_avg.values
        values_api = np.concatenate((values_api, [values_api[0]]))
        ax.plot(theta, values_api[:-1], 'o-', linewidth=3, label='API Models (Avg)',
               color='#FF6B6B', markersize=8)
        ax.fill(theta, values_api[:-1], alpha=0.25, color='#FF6B6B')

        # Open-source models (average)
        values_os = opensource_avg.values
        values_os = np.concatenate((values_os, [values_os[0]]))
        ax.plot(theta, values_os[:-1], 'o-', linewidth=3, label='Open-Source Models (Avg)',
               color='#4ECDC4', markersize=8)
        ax.fill(theta, values_os[:-1], alpha=0.25, color='#4ECDC4')

        ax.set_varlabels(tasks)
        ax.set_ylim(0, 1.0)
        ax.set_title('KLSBench: API vs Open-Source Model Performance',
                    position=(0.5, 1.1), ha='center', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=12)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'radar_model_type_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def create_top_models_comparison(self):
        """Compare top 5 models by average performance"""
        print("\n📊 Creating top models comparison radar chart...")

        # Calculate average performance
        avg_performance = self.pivot_df.mean(axis=1).sort_values(ascending=False)
        top_5_models = avg_performance.head(5).index

        # Prepare data
        tasks = ['Classification', 'Retrieval', 'Punctuation', 'NLI', 'Translation']
        num_vars = len(tasks)

        theta = radar_factory(num_vars, frame='polygon')

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='radar'))

        # Color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        # Plot top 5 models
        for idx, model_name in enumerate(top_5_models):
            values = self.pivot_df.loc[model_name].values
            values = np.concatenate((values, [values[0]]))

            # Shorten model name
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            avg_score = avg_performance[model_name]

            ax.plot(theta, values[:-1], 'o-', linewidth=2.5,
                   label=f'{short_name} (Avg: {avg_score:.3f})',
                   color=colors[idx], markersize=7)
            ax.fill(theta, values[:-1], alpha=0.2, color=colors[idx])

        ax.set_varlabels(tasks)
        ax.set_ylim(0, 1.0)
        ax.set_title('KLSBench: Top 5 Models Performance Comparison',
                    position=(0.5, 1.1), ha='center', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'radar_top5_models.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def create_individual_model_cards(self):
        """Create individual radar chart for each model"""
        print("\n📊 Creating individual model cards...")

        tasks = ['Classification', 'Retrieval', 'Punctuation', 'NLI', 'Translation']
        num_vars = len(tasks)
        theta = radar_factory(num_vars, frame='polygon')

        for model_name, row in self.pivot_df.iterrows():
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))

            values = row.values
            values = np.concatenate((values, [values[0]]))

            # Color based on average performance
            avg_score = row.mean()
            color = plt.cm.RdYlGn(avg_score)

            ax.plot(theta, values[:-1], 'o-', linewidth=3, color=color, markersize=8)
            ax.fill(theta, values[:-1], alpha=0.3, color=color)

            ax.set_varlabels(tasks)
            ax.set_ylim(0, 1.0)

            # Shorten model name for title
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            ax.set_title(f'{short_name}\nAverage Score: {avg_score:.3f}',
                        position=(0.5, 1.1), ha='center', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add score annotations
            for angle, value, task in zip(theta, values[:-1], tasks):
                ax.text(angle, value + 0.05, f'{value:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Safe filename
            safe_name = model_name.replace('/', '_').replace(' ', '_')
            output_path = self.output_dir / f'radar_individual_{safe_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  ✓ Saved {len(self.pivot_df)} individual model cards")

    def create_small_multiples_grid(self):
        """Create a grid of small radar charts (all models)"""
        print("\n📊 Creating small multiples grid...")

        tasks = ['Cls', 'Ret', 'Punc', 'NLI', 'Trans']  # Abbreviated
        num_vars = len(tasks)
        theta = radar_factory(num_vars, frame='polygon')

        # Calculate grid dimensions
        n_models = len(self.pivot_df)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(16, 5 * n_rows))

        for idx, (model_name, row) in enumerate(self.pivot_df.iterrows()):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='radar')

            values = row.values
            values = np.concatenate((values, [values[0]]))

            # Color based on average
            avg_score = row.mean()
            color = plt.cm.RdYlGn(avg_score)

            ax.plot(theta, values[:-1], 'o-', linewidth=2, color=color, markersize=5)
            ax.fill(theta, values[:-1], alpha=0.25, color=color)

            ax.set_varlabels(tasks)
            ax.set_ylim(0, 1.0)

            # Shorten model name
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            if len(short_name) > 20:
                short_name = short_name[:17] + '...'

            ax.set_title(f'{short_name}\n({avg_score:.3f})',
                        fontsize=10, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle('KLSBench: All Models Performance Overview',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        output_path = self.output_dir / 'radar_small_multiples_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def create_performance_summary_table(self):
        """Create a summary table with radar chart visualization"""
        print("\n📊 Creating performance summary table...")

        # Calculate statistics
        summary = pd.DataFrame({
            'Model': self.pivot_df.index,
            'Average': self.pivot_df.mean(axis=1),
            'Classification': self.pivot_df['classification'],
            'Retrieval': self.pivot_df['retrieval'],
            'Punctuation': self.pivot_df['punctuation'],
            'NLI': self.pivot_df['nli'],
            'Translation': self.pivot_df['translation'],
        }).sort_values('Average', ascending=False)

        # Shorten model names
        summary['Model'] = summary['Model'].apply(
            lambda x: x.split('/')[-1] if '/' in x else x
        )

        # Save as CSV
        output_csv = self.output_dir / 'radar_performance_summary.csv'
        summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved: {output_csv}")

        return summary


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Radar Charts for Model Performance')
    parser.add_argument('--results-csv', type=str,
                       default='/Users/songhune/Workspace/korean_eda/results/aggregated/aggregated_summary.csv',
                       help='Path to aggregated results CSV')
    parser.add_argument('--output-dir', type=str,
                       default='../../results/figures',
                       help='Output directory')

    args = parser.parse_args()

    # Resolve output path
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'

    # Generate radar charts
    generator = RadarChartGenerator(args.results_csv, str(output_dir))
    generator.generate_all_radar_charts()
    generator.create_performance_summary_table()

    print("\n" + "="*70)
    print("✅ Radar chart generation completed!")
    print(f"📁 All outputs saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
