#!/usr/bin/env python3
"""
Generate scaling plots and analysis from training results.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(N, a, alpha, c):
    """Power law function: L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_power_law(param_counts, losses):
    """Fit power law to scaling data."""
    # Initial guess
    p0 = [1.0, 0.1, 0.0]
    
    try:
        popt, pcov = curve_fit(power_law, param_counts, losses, p0=p0, maxfev=10000)
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))
        return a, alpha, c, perr
    except Exception as e:
        print(f"Warning: Power law fitting failed: {e}")
        return None, None, None, None


def load_results(results_dir: Path):
    """Load all training results."""
    results_file = results_dir / 'scaling_study_results.json'
    
    if not results_file.exists():
        # Try loading individual model results
        results = {}
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                results_path = model_dir / 'training_results.json'
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        model_results = json.load(f)
                        model_name = model_dir.name
                        results[model_name] = model_results
        return results
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data.get('results', {})


def plot_scaling(results_dir: Path, output_path: Path):
    """Create scaling plot with power law fit."""
    results = load_results(results_dir)
    
    if len(results) == 0:
        print("ERROR: No results found!")
        return
    
    # Extract data
    model_names = []
    param_counts = []
    val_losses = []
    train_losses = []
    
    for name, data in sorted(results.items()):
        if 'epochs' in data and len(data['epochs']) > 0:
            epoch_data = data['epochs'][0]
            params = data.get('model_params', 0)
            val_loss = epoch_data.get('val_loss', float('inf'))
            train_loss = epoch_data.get('train_loss', float('inf'))
            
            if params > 0 and val_loss < float('inf'):
                model_names.append(name)
                param_counts.append(params)
                val_losses.append(val_loss)
                train_losses.append(train_loss)
    
    if len(param_counts) == 0:
        print("ERROR: No valid data points found!")
        return
    
    param_counts = np.array(param_counts)
    val_losses = np.array(val_losses)
    train_losses = np.array(train_losses)
    
    # Fit power law
    a, alpha, c, perr = fit_power_law(param_counts, val_losses)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(param_counts / 1e6, val_losses, s=100, alpha=0.7, 
               label='Validation Loss', color='blue', zorder=3)
    
    # Plot power law fit
    if alpha is not None:
        N_fit = np.logspace(np.log10(param_counts.min()), 
                           np.log10(param_counts.max()), 100)
        L_fit = power_law(N_fit, a, alpha, c)
        ax.plot(N_fit / 1e6, L_fit, 'r--', linewidth=2, 
               label=f'Power Law Fit (α={alpha:.3f}±{perr[1]:.3f})', zorder=2)
    
    # Annotate points
    for i, name in enumerate(model_names):
        ax.annotate(name.upper(), 
                   (param_counts[i] / 1e6, val_losses[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Million Parameters)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Transformer Scaling Law: Validation Loss vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scaling plot saved to: {output_path}")
    
    # Print summary
    print(f"\nScaling Analysis:")
    print(f"  Models: {len(model_names)}")
    print(f"  Parameter range: {param_counts.min()/1e6:.2f}M - {param_counts.max()/1e6:.2f}M")
    print(f"  Loss range: {val_losses.min():.4f} - {val_losses.max():.4f}")
    if alpha is not None:
        print(f"  Scaling exponent (α): {alpha:.4f} ± {perr[1]:.4f}")
        print(f"  Power law: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")


def plot_training_curves(results_dir: Path, output_path: Path):
    """Plot training loss curves for all models."""
    results = load_results(results_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, data in sorted(results.items()):
        metrics_path = Path(results_dir) / name / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            if 'step' in metrics and 'train_loss' in metrics:
                steps = metrics['step']
                losses = metrics['train_loss']
                ax.plot(steps, losses, label=name.upper(), linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Curves for All Models', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves plot saved to: {output_path}")


def create_summary_table(results_dir: Path, output_path: Path):
    """Create summary table of model architectures and statistics."""
    results = load_results(results_dir)
    
    table_data = []
    
    for name, data in sorted(results.items()):
        if 'epochs' in data and len(data['epochs']) > 0:
            epoch_data = data['epochs'][0]
            model_config = data.get('config', {}).get('model', {})
            
            row = {
                'Model': name.upper(),
                'Params (M)': f"{data.get('model_params', 0) / 1e6:.2f}",
                'd_model': model_config.get('d_model', 'N/A'),
                'n_layers': model_config.get('n_layers', 'N/A'),
                'n_heads': model_config.get('n_heads', 'N/A'),
                'd_ff': model_config.get('d_ff', 'N/A'),
                'Val Loss': f"{epoch_data.get('val_loss', 0):.4f}",
                'Train Loss': f"{epoch_data.get('train_loss', 0):.4f}",
                'Time (s)': f"{epoch_data.get('epoch_time', 0):.1f}",
                'GPU Mem (MB)': f"{epoch_data.get('gpu_memory_mb', 0):.0f}"
            }
            table_data.append(row)
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    # Print table
    print(f"\n{'='*100}")
    print("Model Architecture and Training Statistics")
    print(f"{'='*100}")
    print(f"{'Model':<10} {'Params':<10} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'d_ff':<8} {'Val Loss':<10} {'Time(s)':<10}")
    print("-" * 100)
    
    for row in table_data:
        print(f"{row['Model']:<10} {row['Params (M)']:<10} {row['d_model']:<8} "
              f"{row['n_layers']:<8} {row['n_heads']:<8} {row['d_ff']:<8} "
              f"{row['Val Loss']:<10} {row['Time (s)']:<10}")
    
    print(f"{'='*100}")
    print(f"Summary table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate scaling plots and analysis')
    parser.add_argument('--results-dir', type=str, default='outputs/scaling_study',
                       help='Directory containing training results')
    parser.add_argument('--output-dir', type=str, default='outputs/scaling_study/analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    print("Generating scaling analysis...")
    
    # Create scaling plot
    plot_scaling(results_dir, output_dir / 'scaling_plot.png')
    
    # Create training curves
    plot_training_curves(results_dir, output_dir / 'training_curves.png')
    
    # Create summary table
    create_summary_table(results_dir, output_dir / 'summary_table.json')
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


