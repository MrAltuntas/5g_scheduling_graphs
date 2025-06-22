import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configuration
RESULTS_DIR = 'results'
OUTPUT_DIR = 'analysis_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ==================================================
# 1. Fitness Progression Charts (Per Algorithm/Config)
# ==================================================
def plot_fitness_progression():
    algorithms = ['harmony', 'ant_colony']
    param_sets = ['1', '2', '3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors for each param set

    for algo in algorithms:
        plt.figure(figsize=(10, 6))
        plt.title(f'{algo.replace("_", " ").title()} Search - Fitness Progression', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.grid(True, alpha=0.3)

        for i, pset in enumerate(param_sets):
            filename = f"{algo}_fitness_iterations_paramset_{pset}.csv"
            filepath = os.path.join(RESULTS_DIR, filename)

            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            # Read data
            df = pd.read_csv(filepath)
            iterations = df['Iteration']

            # Process retest columns
            retest_cols = [col for col in df.columns if col.startswith('Retest')]
            avg_values = df[retest_cols].mean(axis=1)
            min_values = df[retest_cols].min(axis=1)
            max_values = df[retest_cols].max(axis=1)

            # Plot individual retests as faint lines
            for col in retest_cols:
                plt.plot(iterations, df[col],
                         color=colors[i],
                         alpha=0.1,
                         linewidth=0.8)

            # Plot average line with confidence interval
            plt.plot(iterations, avg_values,
                     color=colors[i],
                     linewidth=2.5,
                     label=f'Config {pset} (Avg)')
            plt.fill_between(iterations, min_values, max_values,
                             color=colors[i],
                             alpha=0.15)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{algo}_fitness_progression.png'))
        plt.close()


# ==================================================
# 2. Algorithm Comparison (Fitness & Time)
# ==================================================
def plot_algorithm_comparison():
    # Read comparison data
    comp_df = pd.read_csv(os.path.join(RESULTS_DIR, 'algorithm_comparison.csv'))

    # Calculate means for each algorithm
    algo_means = comp_df.groupby('Algorithm').agg({
        'AvgFitness': 'mean',
        'BestFitness': 'mean',
        'AvgTime(ms)': 'mean'
    }).reset_index()

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16)

    # Fitness comparison
    sns.barplot(data=algo_means, x='Algorithm', y='AvgFitness',
                palette=['#1f77b4', '#ff7f0e'], ax=ax1, label='Average')
    sns.barplot(data=algo_means, x='Algorithm', y='BestFitness',
                palette=['#1f77b4', '#ff7f0e'], alpha=0.7, ax=ax1, label='Best')

    ax1.set_title('Fitness Comparison')
    ax1.set_ylabel('Fitness Value')
    ax1.set_xlabel('')
    ax1.legend()

    # Time comparison
    sns.barplot(data=algo_means, x='Algorithm', y='AvgTime(ms)',
                palette=['#1f77b4', '#ff7f0e'], ax=ax2)
    ax2.set_title('Computation Time Comparison')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:.0f}s'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'algorithm_comparison.png'))
    plt.close()


# ==================================================
# 3. Configuration Comparison (Per Algorithm)
# ==================================================
def plot_configuration_comparison():
    # Read comparison data
    comp_df = pd.read_csv(os.path.join(RESULTS_DIR, 'algorithm_comparison.csv'))

    for algo in ['Harmony', 'AntColony']:
        algo_df = comp_df[comp_df['Algorithm'] == algo]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{algo} - Configuration Comparison', fontsize=16)

        # Fitness comparison
        sns.barplot(data=algo_df, x='ParamSet', y='AvgFitness',
                    palette='Blues', ax=ax1, label='Average')
        sns.barplot(data=algo_df, x='ParamSet', y='BestFitness',
                    palette='Blues', alpha=0.7, ax=ax1, label='Best')

        ax1.set_title('Fitness by Configuration')
        ax1.set_ylabel('Fitness Value')
        ax1.set_xlabel('Configuration')
        ax1.legend()

        # Time comparison
        sns.barplot(data=algo_df, x='ParamSet', y='AvgTime(ms)',
                    palette='Reds', ax=ax2)
        ax2.set_title('Computation Time by Configuration')
        ax2.set_ylabel('Time (ms)')
        ax2.set_xlabel('Configuration')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:.0f}s'))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{algo.lower()}_config_comparison.png'))
        plt.close()


# ==================================================
# 4. Final Fitness Distribution (Box Plots)
# ==================================================
def plot_final_fitness_distribution():
    algorithms = ['harmony', 'ant_colony']
    param_sets = ['1', '2', '3']

    all_data = []

    for algo in algorithms:
        for pset in param_sets:
            filename = f"{algo}_fitness_iterations_paramset_{pset}.csv"
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
                    'Algorithm': algo.replace('_', ' ').title(),
                    'Config': pset,
                    'Fitness': last_iter[col]
                })

    # Convert to DataFrame
    dist_df = pd.DataFrame(all_data)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=dist_df, x='Algorithm', y='Fitness', hue='Config',
                palette='Set2', showfliers=False)
    sns.stripplot(data=dist_df, x='Algorithm', y='Fitness', hue='Config',
                  palette='dark:black', dodge=True, jitter=0.2, size=4)

    plt.title('Final Fitness Distribution by Algorithm and Configuration', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Fitness Value')
    plt.legend(title='Config', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_fitness_distribution.png'))
    plt.close()


# ==================================================
# 5. Comparative Convergence Analysis
# ==================================================
def plot_convergence_comparison():
    algorithms = ['harmony', 'ant_colony']
    param_sets = ['1', '2', '3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # For param sets
    line_styles = ['-', '--']  # For algorithms

    plt.figure(figsize=(12, 8))

    for algo_idx, algo in enumerate(algorithms):
        for pset_idx, pset in enumerate(param_sets):
            filename = f"{algo}_fitness_iterations_paramset_{pset}.csv"
            filepath = os.path.join(RESULTS_DIR, filename)

            if not os.path.exists(filepath):
                continue

            # Read data
            df = pd.read_csv(filepath)
            iterations = df['Iteration']

            # Calculate average across retests
            retest_cols = [col for col in df.columns if col.startswith('Retest')]
            avg_values = df[retest_cols].mean(axis=1)

            # Plot
            plt.plot(iterations, avg_values,
                     color=colors[pset_idx],
                     linestyle=line_styles[algo_idx],
                     linewidth=2,
                     label=f'{algo.title()} Config {pset}')

    plt.title('Comparative Convergence Analysis', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Fitness', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_comparison.png'))
    plt.close()


# ==================================================
# 6. Computation Time Analysis
# ==================================================
def plot_computation_time_analysis():
    # Read comparison data
    comp_df = pd.read_csv(os.path.join(RESULTS_DIR, 'algorithm_comparison.csv'))

    # Prepare data
    comp_df['Algorithm_Config'] = comp_df['Algorithm'] + ' Config ' + comp_df['ParamSet'].astype(str)

    # Sort by time
    comp_df = comp_df.sort_values('AvgTime(ms)', ascending=False)

    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=comp_df, y='Algorithm_Config', x='AvgTime(ms)',
                     palette='viridis')

    plt.title('Computation Time by Algorithm and Configuration', fontsize=16)
    plt.xlabel('Average Time (ms)', fontsize=12)
    plt.ylabel('')
    plt.xscale('log')

    # Format x-axis labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:.0f}s'))

    # Add value labels
    for i, v in enumerate(comp_df['AvgTime(ms)']):
        ax.text(v, i, f' {v / 1000:.1f}s', color='black', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'computation_time_analysis.png'))
    plt.close()


# ==================================================
# 7. Parameter Analysis (Using Aggregated Results)
# ==================================================
def plot_parameter_analysis():
    # Read Harmony parameters
    harmony_df = pd.read_csv(os.path.join(RESULTS_DIR, 'harmony_aggregated_results.csv'))
    harmony_df['Algorithm'] = 'Harmony'

    # Read Ant Colony parameters
    ant_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ant_colony_aggregated_results.csv'))
    ant_df['Algorithm'] = 'Ant Colony'

    # Combine data
    param_df = pd.concat([harmony_df, ant_df], ignore_index=True)

    # Create parameter labels
    param_df['Config'] = param_df.apply(lambda row:
                                        f"Iter:{row['Iterations']}\n" +
                                        (f"HMS:{row['HMS']}\nHMCR:{row['HMCR']}\nPAR:{row['PAR']}" if row[
                                                                                                          'Algorithm'] == 'Harmony'
                                         else f"Ants:{row['NumAnts']}\nPhero:{row['PheroCoeff']}\nHeur:{row['HeurCoeff']}\nRho:{row['Rho']}\nQ:{row['Q']}"),
                                        axis=1)

    # Plot
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=param_df, x='AvgFitness', y='AvgTime(ms)',
                    hue='Algorithm', size='BestFitness',
                    sizes=(50, 200), alpha=0.8, palette=['#1f77b4', '#ff7f0e'])

    # Add config labels
    for i, row in param_df.iterrows():
        plt.annotate(row['Config'],
                     (row['AvgFitness'], row['AvgTime(ms)']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

    plt.title('Parameter Configuration Analysis', fontsize=16)
    plt.xlabel('Average Fitness', fontsize=12)
    plt.ylabel('Computation Time (ms)', fontsize=12)
    plt.yscale('log')
    plt.yticks([10000, 30000, 100000, 300000], ['10s', '30s', '100s', '300s'])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_analysis.png'))
    plt.close()


# ==================================================
# MAIN EXECUTION
# ==================================================
if __name__ == '__main__':
    print("Generating visualizations...")

    # Create all plots
    plot_fitness_progression()
    plot_algorithm_comparison()
    plot_configuration_comparison()
    plot_final_fitness_distribution()
    plot_convergence_comparison()
    plot_computation_time_analysis()
    plot_parameter_analysis()

    print(f"Visualizations saved to: {OUTPUT_DIR}")
