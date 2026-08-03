"""
Visualization Script for Ablation Study Results
Generates publication-quality plots from ablation JSON data.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def load_all_results(results_dir="data/ablation_results"):
    """Finds and loads ALL ablation_raw_*.json files and merges them."""
    p = Path(results_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")
    
    files = list(p.glob("ablation_raw_*.json"))
    if not files:
        raise FileNotFoundError("No ablation results found in data/ablation_results")
        
    all_data = []
    print(f"Found {len(files)} result files. Merging...")
    
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"Skipping corrupt file {f_path}: {e}")
            
    # Flatten structure
    df = pd.DataFrame(all_data)
    
    # Remove duplicates (keep latest based on some logic, or just first occurence if idential)
    # We deduplicate by ['name', 'subject', 'source']
    initial_len = len(df)
    df = df.drop_duplicates(subset=['name', 'subject', 'source'], keep='last')
    print(f"Merged {initial_len} records -> {len(df)} unique records.")
    
    # Calculate Abs Error
    df['abs_error'] = np.abs(df['pred_bpm'] - df['gt_bpm'])
    df['squared_error'] = (df['pred_bpm'] - df['gt_bpm']) ** 2
    
    return df

def plot_performance_summary(df, output_dir):
    """Generates Bar Chart for MAE across baselines with Literature Benchmarks."""
    # Calculate metrics by baseline
    metrics = df.groupby('name').agg({
        'abs_error': 'mean',
        'squared_error': 'mean'
    }).rename(columns={'abs_error': 'MAE'})
    
    metrics['RMSE'] = np.sqrt(metrics['squared_error'])
    metrics = metrics.reset_index()[['name', 'MAE', 'RMSE']]
    
    # Add Literature Benchmarks (Source: Manne et al. AIR-125 / Related Papers)
    # Using specific markers (*) or color diff to distinguish?
    benchmarks = pd.DataFrame([
        {'name': 'Lit: AIRFlowNet (SOTA)', 'MAE': 2.9, 'RMSE': 3.5},
        {'name': 'Lit: DeepPhys (Adult)', 'MAE': 5.5, 'RMSE': 6.8},
        {'name': 'Lit: Pose+Flow', 'MAE': 3.8, 'RMSE': 4.9}
    ])
    
    combined = pd.concat([metrics, benchmarks], ignore_index=True)
    combined = combined.sort_values('MAE', ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Create palette: gray for Lit, viridis for Ours
    palette = {}
    for name in combined['name']:
        if name == "Full-System":
            palette[name] = "#d32f2f" # Red (Hero) - Material Red 700
        elif name.startswith("Lit:") or "(Real)" in name:
            palette[name] = "#95a5a6" # Gray (Benchmark)
        else:
            palette[name] = "#1976d2" # Blue (Baseline) - Material Blue 700
            
    ax = sns.barplot(x='name', y='MAE', hue='name', data=combined, palette=palette, legend=False)
    
    plt.title('Performance Comparison: MM-RespNet (Full-System) vs. Benchmarks', fontweight='bold', fontsize=14)
    plt.ylabel('Mean Absolute Error (BPM)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add value labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f', padding=3)

    plt.tight_layout()
    plt.savefig(output_dir / "mae_comparison.png", dpi=300)
    print("[v] Saved mae_comparison.png")
    plt.close()

def plot_error_distribution(df, output_dir):
    """Generates Box Plot to show error spread/stability."""
    plt.figure(figsize=(12, 6))
    
    # Sort order by MAE
    order = df.groupby('name')['abs_error'].mean().sort_values(ascending=False).index
    
    sns.boxplot(x='name', y='abs_error', hue='name', data=df, order=order, palette='viridis', legend=False)
    
    plt.title('Error Distribution & Stability Analysis', fontweight='bold', fontsize=14)
    plt.ylabel('Absolute Error (BPM)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution_boxplot.png", dpi=300)
    print("[v] Saved error_distribution_boxplot.png")
    plt.close()

def plot_gt_vs_pred(df, output_dir):
    """Generates Scatter Plot of Ground Truth vs Predicted."""
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot with hue for baselines
    sns.scatterplot(x='gt_bpm', y='pred_bpm', hue='name', data=df, alpha=0.6, s=100)
    
    # Ideal line
    min_val = min(df['gt_bpm'].min(), df['pred_bpm'].min())
    max_val = max(df['gt_bpm'].max(), df['pred_bpm'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)', linewidth=2)
    
    # Calculate Pearson Correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(df['gt_bpm'], df['pred_bpm'])
    plt.text(0.05, 0.95, f'Pearson r = {corr:.2f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12, verticalalignment='top')
    
    plt.title('Ground Truth vs. Predicted BPM', fontweight='bold', fontsize=14)
    plt.ylabel('Predicted BPM', fontsize=12)
    plt.xlabel('Reference (Ground Truth) BPM', fontsize=12)
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "gt_vs_pred_scatter.png", dpi=300)
    print("[v] Saved gt_vs_pred_scatter.png")
    plt.close()

def generate_plots():
    try:
        df = load_all_results()
        
        # Create plots directory
        output_dir = Path("data/ablation_results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating plots for {len(df)} experiments...")
        
        plot_performance_summary(df, output_dir)
        plot_error_distribution(df, output_dir)
        plot_gt_vs_pred(df, output_dir)
        
        print(f"\nAll plots saved to: {output_dir.absolute()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Tip: Run 'python src/run_ablation_study.py' first to generate results.")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    generate_plots()
