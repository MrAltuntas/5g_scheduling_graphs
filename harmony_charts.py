import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configuration
RESULTS_DIR = 'results'
OUTPUT_DIR = 'harmony_charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAM_SETS = ['1', '2', '3']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Set plot styles for maximum readability
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_improved_fitness_progression():
    """Plot fitness progression for each configuration with enhanced readability"""
    for pset in PARAM_SETS:
        plt.figure(figsize=(10, 6))
        filename = f"harmony_fitness_iterations_paramset_{pset}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        # Read data
        df = pd.read_csv(filepath)
        iterations = df['Iteration']

        # Process retest columns
        retest_cols = [col for col in df.columns if col.startswith('Retest')]

        # Calculate statistics
        avg_values = df[retest_cols].mean(axis=1)
        min_values = df[retest_cols].min(axis=1)
        max_values = df[retest_cols].max(axis=1)

        # Create a clean plot with minimal lines
        # Plot min/max range as a shaded area
        plt.fill_between(iterations, min_values, max_values,
                         color=COLORS[PARAM_SETS.index(pset)],
                         alpha=0.15, label='Min/Max Range')

        # Plot average line
        plt.plot(iterations, avg_values,
                 color=COLORS[PARAM_SETS.index(pset)],
                 linewidth=2.5, label='Average Fitness')

        # Mark final value clearly
        final_avg = avg_values.iloc[-1]
        plt.scatter(iterations.max(), final_avg,
                    color='darkred', s=80, zorder=5)
        plt.text(iterations.max() + 0.5, final_avg,
                 f'Final: {final_avg:.1f}',
                 va='center', ha='left', fontsize=10)

        # Configure plot
        plt.title(f'Harmony Search - Config {pset} Fitness Progression', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save as PNG
        plt.savefig(os.path.join(OUTPUT_DIR, f'harmony_config_{pset}_fitness.png'))
        plt.close()


def plot_final_fitness_distribution():
    """Plot box plot of final fitness values for each configuration"""
    plt.figure(figsize=(10, 6))
    all_data = []

    for pset in PARAM_SETS:
        filename = f"harmony_fitness_iterations_paramset_{pset}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filepath):
            continue

        # Read data and get last iteration
        df = pd.read_csv(filepath)
        last_iter = df.iloc[-1]

        # Get retest columns
        retest_cols = [col for col in df.columns if col.startswith('Retest')]
        for col in retest_cols:
            all_data.append({
                'Config': pset,
                'Fitness': last_iter[col]
            })

    # Convert to DataFrame
    dist_df = pd.DataFrame(all_data)

    # Create box plot
    sns.boxplot(data=dist_df, x='Config', y='Fitness',
                palette=COLORS, width=0.5, showfliers=False)

    # Add individual points with jitter
    sns.stripplot(data=dist_df, x='Config', y='Fitness',
                  color='black', size=5, alpha=0.7, jitter=0.2)

    # Add mean markers
    means = dist_df.groupby('Config')['Fitness'].mean()
    for i, config in enumerate(PARAM_SETS):
        plt.scatter(i, means[config], marker='s', s=80,
                    color='darkred', label='Mean' if i == 0 else "")

    plt.title('Final Fitness Distribution by Configuration', fontsize=14)
    plt.xlabel('Configuration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'harmony_final_fitness_distribution.png'))
    plt.close()


def plot_convergence_comparison():
    """Plot convergence comparison between configurations"""
    plt.figure(figsize=(10, 6))

    for pset in PARAM_SETS:
        filename = f"harmony_fitness_iterations_paramset_{pset}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filepath):
            continue

        # Read data
        df = pd.read_csv(filepath)
        iterations = df['Iteration']

        # Calculate average across retests
        retest_cols = [col for col in df.columns if col.startswith('Retest')]
        avg_values = df[retest_cols].mean(axis=1)

        # Plot with thicker line
        plt.plot(iterations, avg_values,
                 color=COLORS[PARAM_SETS.index(pset)],
                 linewidth=2.5,
                 label=f'Config {pset}')

        # Mark final value
        final_value = avg_values.iloc[-1]
        plt.scatter(iterations.max(), final_value,
                    color=COLORS[PARAM_SETS.index(pset)], s=80, zorder=5)
        plt.text(iterations.max() + 0.5, final_value, f'{final_value:.1f}',
                 va='center', fontsize=10)

    plt.title('Convergence Comparison', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Average Fitness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'harmony_convergence_comparison.png'))
    plt.close()


def plot_computation_time():
    """Plot computation time analysis for Harmony configurations"""
    # Read aggregated results
    agg_file = os.path.join(RESULTS_DIR, 'harmony_aggregated_results.csv')
    if not os.path.exists(agg_file):
        # Try alternative filename if needed
        agg_file = os.path.join(RESULTS_DIR, 'harmony_aggregated_results.csv')
        if not os.path.exists(agg_file):
            return

    agg_df = pd.read_csv(agg_file)

    # Prepare data
    agg_df['Config'] = PARAM_SETS[:len(agg_df)]

    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=agg_df, x='Config', y='AvgTime(ms)',
                     palette=COLORS, alpha=0.8)

    # Add value labels in seconds
    for i, v in enumerate(agg_df['AvgTime(ms)']):
        ax.text(i, v, f' {v / 1000:.1f}s', color='black',
                ha='center', va='bottom', fontsize=10, weight='bold')

    plt.title('Computation Time by Configuration', fontsize=14)
    plt.xlabel('Configuration')
    plt.ylabel('Average Time (ms)')

    # Format y-axis as seconds
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:.0f}s'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'harmony_computation_time.png'))
    plt.close()


def plot_parameter_performance():
    """Plot parameter performance comparison"""
    # Read aggregated results
    agg_file = os.path.join(RESULTS_DIR, 'harmony_aggregated_results.csv')
    if not os.path.exists(agg_file):
        return

    agg_df = pd.read_csv(agg_file)
    agg_df['Config'] = PARAM_SETS[:len(agg_df)]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot fitness values
    ax1 = plt.gca()
    ax1.bar(agg_df['Config'], agg_df['AvgFitness'],
            color=COLORS, alpha=0.6, label='Average Fitness')
    ax1.scatter(agg_df['Config'], agg_df['BestFitness'],
                color='darkred', s=100, zorder=5, label='Best Fitness')

    # Add value labels
    for i, row in agg_df.iterrows():
        ax1.text(i, row['AvgFitness'], f'{row["AvgFitness"]:.1f}',
                 ha='center', va='bottom', fontsize=10, weight='bold')
        ax1.text(i, row['BestFitness'], f'{row["BestFitness"]:.1f}',
                 ha='center', va='bottom', fontsize=10, weight='bold')

    ax1.set_ylabel('Fitness Value')
    ax1.legend(loc='upper left')

    # Create second y-axis for time
    ax2 = ax1.twinx()
    ax2.plot(agg_df['Config'], agg_df['AvgTime(ms)'] / 1000,
             color='purple', marker='o', markersize=8,
             linewidth=2, label='Time (s)')

    # Add time value labels
    for i, v in enumerate(agg_df['AvgTime(ms)']):
        ax2.text(i, v / 1000, f'{v / 1000:.1f}s',
                 ha='center', va='bottom', fontsize=10, weight='bold')

    ax2.set_ylabel('Computation Time (seconds)')
    ax2.legend(loc='upper right')

    plt.title('Performance Comparison by Configuration', fontsize=14)
    plt.xlabel('Configuration')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'harmony_performance_comparison.png'))
    plt.close()


def main():
    """Generate all Harmony Search charts"""
    print("Generating Harmony Search charts...")

    # Create all plots
    plot_improved_fitness_progression()
    plot_final_fitness_distribution()
    plot_convergence_comparison()
    plot_computation_time()
    plot_parameter_performance()

    print(f"Charts saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()