#!/usr/bin/env python3
"""
Scaling Study: Train multiple transformer and RNN models of varying sizes.
"""

import sys
import json
import time
import gc
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.optim import AdamW

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import MusicTransformer
from models.rnn import MusicLSTM
from utils.data_loader import MusicDataLoader
from utils.training import train_one_epoch, evaluate, get_lr_schedule


# Model configurations for scaling study
TRANSFORMER_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 2,
        'd_ff': 512,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 1e6  # ~1M
    },
    'small': {
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'd_ff': 1024,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 5e6  # ~5M
    },
    'medium': {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 2048,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 20e6  # ~20M
    },
    'large': {
        'd_model': 768,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 3072,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 50e6  # ~50M
    },
    'xl': {
        'd_model': 1024,
        'n_layers': 12,
        'n_heads': 12,
        'd_ff': 4096,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 100e6  # ~100M+
    }
}

# RNN/LSTM configurations (matching parameter counts)
RNN_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'n_layers': 2,
        'dropout': 0.1,
        'max_seq_length': 5000,
        'target_params': 1e6
    },
    'small': {
        'd_model': 256,
        'n_layers': 3,
        'dropout': 0.1,
        'max_seq_length': 5000,
        'target_params': 5e6
    },
    'medium': {
        'd_model': 512,
        'n_layers': 4,
        'dropout': 0.1,
        'max_seq_length': 5000,
        'target_params': 20e6
    },
    'large': {
        'd_model': 768,
        'n_layers': 5,
        'dropout': 0.1,
        'max_seq_length': 5000,
        'target_params': 50e6
    }
}


def run_transformer_scaling_study(
    tokenizer,
    train_path,
    val_path,
    output_dir,
    device,
    learning_rate=3e-4,
    num_epochs=1,
    batch_size_tokens=50000,
    log_interval=100
):
    """Run transformer scaling study."""
    print("="*60)
    print("TRANSFORMER SCALING STUDY")
    print("="*60)
    print(f"Training {len(TRANSFORMER_CONFIGS)} transformer models")
    print(f"Each model will train for {num_epochs} epoch(s)")
    print(f"Consistent hyperparameters: LR={learning_rate}, Batch={batch_size_tokens:,} tokens")
    print("="*60)
    
    transformer_results = []
    
    for model_name, config in TRANSFORMER_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Transformer")
        print(f"{'='*60}")
        
        # Initialize model
        model = MusicTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout']
        ).to(device)
        
        num_params = model.count_parameters()
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"Target: {config['target_params']/1e6:.1f}M")
        
        # Create data loaders
        train_loader = MusicDataLoader(
            train_path,
            batch_size_tokens=batch_size_tokens,
            max_seq_length=config['max_seq_length'],
            shuffle=True
        )
        
        val_loader = MusicDataLoader(
            val_path,
            batch_size_tokens=batch_size_tokens,
            max_seq_length=config['max_seq_length'],
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        estimated_steps = len(train_loader) if hasattr(train_loader, '__len__') else 1000
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = get_lr_schedule(optimizer, estimated_steps, warmup_steps=0)
        
        # Train
        start_time = time.time()
        train_losses = []
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                log_interval=log_interval
            )
            train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        training_time = time.time() - start_time
        
        # Get GPU memory usage
        gpu_memory_mb = None
        if device.type == 'cuda':
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            torch.cuda.reset_peak_memory_stats(device)
        
        # Store results
        result = {
            'model_name': model_name,
            'architecture': 'transformer',
            'num_parameters': num_params,
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_loss,
            'training_time_seconds': training_time,
            'gpu_memory_mb': gpu_memory_mb,
            'config': config
        }
        transformer_results.append(result)
        
        print(f"\n{model_name.upper()} Results:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Train Loss: {train_losses[-1]:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Training Time: {training_time/60:.1f} minutes")
        if gpu_memory_mb:
            print(f"  GPU Memory: {gpu_memory_mb:.1f} MB")
        
        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        gc.collect()
    
    print(f"\n{'='*60}")
    print("TRANSFORMER SCALING STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Trained {len(transformer_results)} models")
    for r in transformer_results:
        print(f"  {r['model_name']:8s}: {r['num_parameters']/1e6:6.2f}M params, Val Loss: {r['val_loss']:.4f}")
    
    return transformer_results


def run_rnn_scaling_study(
    tokenizer,
    train_path,
    val_path,
    output_dir,
    device,
    learning_rate=3e-4,
    num_epochs=1,
    batch_size_tokens=50000,
    log_interval=100
):
    """Run RNN/LSTM scaling study."""
    print("="*60)
    print("RNN/LSTM SCALING STUDY")
    print("="*60)
    print(f"Training {len(RNN_CONFIGS)} LSTM models")
    print(f"Each model will train for {num_epochs} epoch(s)")
    print(f"Consistent hyperparameters: LR={learning_rate}, Batch={batch_size_tokens:,} tokens")
    print("="*60)
    
    rnn_results = []
    
    for model_name, config in RNN_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} LSTM")
        print(f"{'='*60}")
        
        # Initialize model
        model = MusicLSTM(
            vocab_size=tokenizer.vocab_size,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            max_seq_length=config['max_seq_length']
        ).to(device)
        
        num_params = model.count_parameters()
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"Target: {config['target_params']/1e6:.1f}M")
        
        # Create data loaders
        train_loader = MusicDataLoader(
            train_path,
            batch_size_tokens=batch_size_tokens,
            max_seq_length=config['max_seq_length'],
            shuffle=True
        )
        
        val_loader = MusicDataLoader(
            val_path,
            batch_size_tokens=batch_size_tokens,
            max_seq_length=config['max_seq_length'],
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        estimated_steps = len(train_loader) if hasattr(train_loader, '__len__') else 1000
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = get_lr_schedule(optimizer, estimated_steps, warmup_steps=0)
        
        # Train
        start_time = time.time()
        train_losses = []
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                log_interval=log_interval
            )
            train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        training_time = time.time() - start_time
        
        # Get GPU memory usage
        gpu_memory_mb = None
        if device.type == 'cuda':
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            torch.cuda.reset_peak_memory_stats(device)
        
        # Store results
        result = {
            'model_name': model_name,
            'architecture': 'rnn',
            'num_parameters': num_params,
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_loss,
            'training_time_seconds': training_time,
            'gpu_memory_mb': gpu_memory_mb,
            'config': config
        }
        rnn_results.append(result)
        
        print(f"\n{model_name.upper()} Results:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Train Loss: {train_losses[-1]:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Training Time: {training_time/60:.1f} minutes")
        if gpu_memory_mb:
            print(f"  GPU Memory: {gpu_memory_mb:.1f} MB")
        
        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        gc.collect()
    
    print(f"\n{'='*60}")
    print("RNN SCALING STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Trained {len(rnn_results)} models")
    for r in rnn_results:
        print(f"  {r['model_name']:8s}: {r['num_parameters']/1e6:6.2f}M params, Val Loss: {r['val_loss']:.4f}")
    
    return rnn_results


if __name__ == "__main__":
    import argparse
    from utils.data_loader import load_tokenizer
    
    parser = argparse.ArgumentParser(description="Run scaling study")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--output-dir", type=str, default="outputs/scaling_study",
                       help="Output directory for results")
    parser.add_argument("--architecture", type=str, choices=['transformer', 'rnn', 'both'],
                       default='both', help="Which architecture(s) to train")
    parser.add_argument("--num-epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch-size-tokens", type=int, default=50000,
                       help="Batch size in tokens")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = data_dir / "tokenizer.pkl"
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Data paths
    train_path = data_dir / "tokenized" / "train" / "data.json"
    val_path = data_dir / "tokenized" / "val" / "data.json"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run scaling studies
    scaling_results = {'transformer': [], 'rnn': []}
    
    if args.architecture in ['transformer', 'both']:
        transformer_results = run_transformer_scaling_study(
            tokenizer, train_path, val_path, output_dir, device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size_tokens=args.batch_size_tokens
        )
        scaling_results['transformer'] = transformer_results
    
    if args.architecture in ['rnn', 'both']:
        rnn_results = run_rnn_scaling_study(
            tokenizer, train_path, val_path, output_dir, device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size_tokens=args.batch_size_tokens
        )
        scaling_results['rnn'] = rnn_results
    
    # Save results
    results_path = output_dir / "scaling_results.json"
    with open(results_path, 'w') as f:
        json.dump(scaling_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")

