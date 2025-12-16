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
    # First try loading from scaling_results.json (notebook format)
    scaling_results_file = results_dir / 'scaling_results.json'
    if scaling_results_file.exists():
        with open(scaling_results_file, 'r') as f:
            data = json.load(f)
            # Return in format compatible with old code
            results = {}
            if 'transformer' in data:
                for r in data['transformer']:
                    results[f"transformer_{r['model_name']}"] = {
                        'model_params': r['num_parameters'],
                        'epochs': [{
                            'val_loss': r['val_loss'],
                            'train_loss': r.get('train_loss', 0)
                        }]
                    }
            if 'rnn' in data:
                for r in data['rnn']:
                    results[f"rnn_{r['model_name']}"] = {
                        'model_params': r['num_parameters'],
                        'epochs': [{
                            'val_loss': r['val_loss'],
                            'train_loss': r.get('train_loss', 0)
                        }]
                    }
            return results
    
    # Try loading from scaling_study_results.json
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
    """Create scaling plot with power law fit for both transformers and RNNs."""
    # Try loading from scaling_results.json (notebook format)
    scaling_results_file = results_dir / 'scaling_results.json'
    
    transformer_results = []
    rnn_results = []
    
    if scaling_results_file.exists():
        with open(scaling_results_file, 'r') as f:
            data = json.load(f)
            transformer_results = data.get('transformer', [])
            rnn_results = data.get('rnn', [])
    else:
        # Fallback to old format
        results = load_results(results_dir)
        
        if len(results) == 0:
            print("ERROR: No results found!")
            return
        
        # Extract data
        for name, data in sorted(results.items()):
            if 'epochs' in data and len(data['epochs']) > 0:
                epoch_data = data['epochs'][0]
                params = data.get('model_params', 0)
                val_loss = epoch_data.get('val_loss', float('inf'))
                
                if params > 0 and val_loss < float('inf'):
                    result = {
                        'model_name': name,
                        'num_parameters': params,
                        'val_loss': val_loss,
                        'train_loss': epoch_data.get('train_loss', 0)
                    }
                    if 'transformer' in name.lower():
                        transformer_results.append(result)
                    elif 'rnn' in name.lower() or 'lstm' in name.lower():
                        rnn_results.append(result)
    
    if len(transformer_results) == 0 and len(rnn_results) == 0:
        print("ERROR: No valid data points found!")
        return
    
    # Prepare data for plotting
    transformer_params = np.array([r['num_parameters'] for r in transformer_results]) if transformer_results else np.array([])
    transformer_losses = np.array([r['val_loss'] for r in transformer_results]) if transformer_results else np.array([])
    
    rnn_params = np.array([r['num_parameters'] for r in rnn_results]) if rnn_results else np.array([])
    rnn_losses = np.array([r['val_loss'] for r in rnn_results]) if rnn_results else np.array([])
    
    # Fit power laws
    transformer_fit = fit_power_law(transformer_params, transformer_losses) if len(transformer_params) > 0 else (None, None, None, None)
    rnn_fit = fit_power_law(rnn_params, rnn_losses) if len(rnn_params) > 0 else (None, None, None, None)
    
    # Create combined comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Individual scaling plots
    ax1.set_xscale('log')
    if len(transformer_params) > 0:
        ax1.scatter(transformer_params / 1e6, transformer_losses, 
                   s=150, alpha=0.7, label='Transformer', color='blue', zorder=3, marker='o')
    if len(rnn_params) > 0:
        ax1.scatter(rnn_params / 1e6, rnn_losses, 
                   s=150, alpha=0.7, label='RNN/LSTM', color='red', zorder=3, marker='s')
    
    # Plot power law fits
    if transformer_fit[0] is not None:
        a, alpha, c, perr = transformer_fit
        N_fit = np.logspace(np.log10(transformer_params.min()), 
                           np.log10(transformer_params.max()), 100)
        L_fit = power_law(N_fit, a, alpha, c)
        ax1.plot(N_fit / 1e6, L_fit, 'b--', linewidth=2, 
                label=f'Transformer Fit (α={alpha:.3f}±{perr[1]:.3f})', zorder=2)
    
    if rnn_fit[0] is not None:
        a, alpha, c, perr = rnn_fit
        N_fit = np.logspace(np.log10(rnn_params.min()), 
                           np.log10(rnn_params.max()), 100)
        L_fit = power_law(N_fit, a, alpha, c)
        ax1.plot(N_fit / 1e6, L_fit, 'r--', linewidth=2, 
                label=f'RNN Fit (α={alpha:.3f}±{perr[1]:.3f})', zorder=2)
    
    # Annotate points
    for r in transformer_results:
        ax1.annotate(r['model_name'].upper(), 
                    (r['num_parameters']/1e6, r['val_loss']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for r in rnn_results:
        ax1.annotate(r['model_name'].upper(), 
                    (r['num_parameters']/1e6, r['val_loss']),
                    xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Model Size (Million Parameters)', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Scaling Laws: Validation Loss vs Model Size', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    
    # Plot 2: Comparison (overlay)
    ax2.set_xscale('log')
    if len(transformer_params) > 0:
        ax2.scatter(transformer_params / 1e6, transformer_losses, 
                   s=150, alpha=0.7, label='Transformer', color='blue', zorder=3, marker='o')
    if len(rnn_params) > 0:
        ax2.scatter(rnn_params / 1e6, rnn_losses, 
                   s=150, alpha=0.7, label='RNN/LSTM', color='red', zorder=3, marker='s')
    
    if transformer_fit[0] is not None:
        a, alpha, c, perr = transformer_fit
        N_fit = np.logspace(np.log10(transformer_params.min()), 
                           np.log10(transformer_params.max()), 100)
        L_fit = power_law(N_fit, a, alpha, c)
        ax2.plot(N_fit / 1e6, L_fit, 'b--', linewidth=2, zorder=2)
    
    if rnn_fit[0] is not None:
        a, alpha, c, perr = rnn_fit
        N_fit = np.logspace(np.log10(rnn_params.min()), 
                           np.log10(rnn_params.max()), 100)
        L_fit = power_law(N_fit, a, alpha, c)
        ax2.plot(N_fit / 1e6, L_fit, 'r--', linewidth=2, zorder=2)
    
    ax2.set_xlabel('Model Size (Million Parameters)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Architecture Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scaling plots saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"\nTransformer Models: {len(transformer_results)}")
    if len(transformer_params) > 0:
        print(f"  Parameter range: {transformer_params.min()/1e6:.2f}M - {transformer_params.max()/1e6:.2f}M")
        print(f"  Loss range: {transformer_losses.min():.4f} - {transformer_losses.max():.4f}")
        if transformer_fit[0] is not None:
            a, alpha, c, perr = transformer_fit
            print(f"  Scaling exponent (α): {alpha:.4f} ± {perr[1]:.4f}")
            print(f"  Power law: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
    
    print(f"\nRNN/LSTM Models: {len(rnn_results)}")
    if len(rnn_params) > 0:
        print(f"  Parameter range: {rnn_params.min()/1e6:.2f}M - {rnn_params.max()/1e6:.2f}M")
        print(f"  Loss range: {rnn_losses.min():.4f} - {rnn_losses.max():.4f}")
        if rnn_fit[0] is not None:
            a, alpha, c, perr = rnn_fit
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


