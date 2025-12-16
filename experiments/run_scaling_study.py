#!/usr/bin/env python3
"""
Run scaling study: train multiple model sizes and collect metrics.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_training(config_path: Path, output_base_dir: Path, data_dir: Path,
                batch_size_tokens: int = 512000, learning_rate: float = 3e-4):
    """Run training for a single model configuration."""
    config_name = config_path.stem
    print(f"\n{'='*80}")
    print(f"Training {config_name.upper()} model")
    print(f"{'='*80}\n")
    
    # Run training script
    # Get the project root directory (parent of experiments/)
    project_root = Path(__file__).parent.parent
    train_script = project_root / 'experiments' / 'train.py'
    
    cmd = [
        sys.executable, str(train_script),
        '--config', str(config_path),
        '--data-dir', str(data_dir),
        '--output-dir', str(output_base_dir),
        '--batch-size-tokens', str(batch_size_tokens),
        '--learning-rate', str(learning_rate),
        '--max-epochs', '1',
        '--log-interval', '50'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {config_name}")
        return None
    
    # Load results
    output_dir = output_base_dir / config_name
    results_path = output_dir / 'training_results.json'
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    else:
        print(f"WARNING: Results file not found: {results_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run scaling study')
    parser.add_argument('--configs-dir', type=str, default='configs',
                       help='Directory containing model configs')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/scaling_study',
                       help='Output directory for all results')
    parser.add_argument('--batch-size-tokens', type=int, default=512000,
                       help='Batch size in tokens (default: 512K)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['tiny', 'small', 'medium', 'large', 'xl'],
                       help='Model sizes to train (default: all)')
    
    args = parser.parse_args()
    
    configs_dir = Path(args.configs_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find config files
    config_files = []
    for model_name in args.models:
        config_path = configs_dir / f"{model_name}.yaml"
        if config_path.exists():
            config_files.append(config_path)
        else:
            print(f"WARNING: Config not found: {config_path}")
    
    if len(config_files) == 0:
        print("ERROR: No valid config files found!")
        return 1
    
    print(f"\nScaling Study Configuration:")
    print(f"  Models to train: {len(config_files)}")
    print(f"  Batch size (tokens): {args.batch_size_tokens:,}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {output_dir}")
    
    # Run training for each model
    all_results = {}
    
    for config_path in config_files:
        config_name = config_path.stem
        results = run_training(
            config_path, output_dir, data_dir,
            args.batch_size_tokens, args.learning_rate
        )
        
        if results is not None:
            all_results[config_name] = results
    
    # Save combined results
    study_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size_tokens': args.batch_size_tokens,
            'learning_rate': args.learning_rate,
            'models_trained': list(all_results.keys())
        },
        'results': all_results
    }
    
    results_file = output_dir / 'scaling_study_results.json'
    with open(results_file, 'w') as f:
        json.dump(study_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Scaling Study Complete")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"\nModels trained: {len(all_results)}")
    for name, results in all_results.items():
        if 'epochs' in results and len(results['epochs']) > 0:
            val_loss = results['epochs'][0]['val_loss']
            params = results.get('model_params', 0)
            print(f"  {name:10s}: {params/1e6:6.2f}M params, val_loss={val_loss:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


