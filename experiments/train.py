#!/usr/bin/env python3
"""
Training script for MusicTransformer models.
Maintains consistent hyperparameters across all model sizes.
"""

import sys
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import MusicTransformer
from utils.data_loader import MusicDataLoader, load_tokenizer
from utils.metrics import TrainingMetrics


def get_lr_schedule(optimizer, num_steps, warmup_steps=0):
    """Create cosine annealing learning rate schedule."""
    if warmup_steps > 0:
        # Linear warmup + cosine annealing
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Cosine annealing only
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    return scheduler


def train_one_epoch(model, train_loader, optimizer, scheduler, device, 
                   metrics: TrainingMetrics, log_interval=100):
    """Train model for one epoch."""
    model.train()
    metrics.start_epoch()
    
    total_loss = 0.0
    num_steps = 0
    
    for step, (input_ids, target_ids) in enumerate(train_loader):
        # Move to device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Clamp token IDs to valid range [0, vocab_size-1]
        # Replace padding tokens (-1) with 0 (or use a proper padding token)
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        
        # Replace -1 (padding) with 0 for input_ids, keep -1 for target_ids (will be ignored in loss)
        input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
        
        # Forward pass
        logits, loss = model(input_ids, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        num_steps += 1
        
        # Get GPU memory usage
        gpu_memory_mb = None
        if device.type == 'cuda':
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        
        # Log metrics
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        metrics.log_step(step, loss.item(), lr=current_lr, gpu_memory_mb=gpu_memory_mb)
        
        # Print progress
        if step % log_interval == 0:
            avg_loss = total_loss / num_steps
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    
    avg_loss = total_loss / num_steps
    epoch_time = metrics.get_epoch_time()
    print(f"\nEpoch complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_steps = 0
    
    for input_ids, target_ids in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Clamp token IDs to valid range [0, vocab_size-1]
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        
        # Replace -1 (padding) with 0 for input_ids
        input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
        
        logits, loss = model(input_ids, target_ids)
        total_loss += loss.item()
        num_steps += 1
    
    avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train MusicTransformer')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config YAML file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Path to output directory')
    parser.add_argument('--batch-size-tokens', type=int, default=512000,
                       help='Batch size in tokens (default: 512K)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='Number of warmup steps (default: 0)')
    parser.add_argument('--max-epochs', type=int, default=1,
                       help='Number of epochs to train (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Log interval in steps (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    print(f"\nModel Configuration:")
    print(f"  Name: {config.get('name', 'unknown')}")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  n_layers: {model_config['n_layers']}")
    print(f"  n_heads: {model_config['n_heads']}")
    print(f"  d_ff: {model_config['d_ff']}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / 'tokenizer.pkl'
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        max_seq_length=model_config.get('max_seq_length', 5000),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Create data loaders
    # Try both .json and .jsonl extensions
    train_path = data_dir / 'tokenized' / 'train' / 'data.json'
    if not train_path.exists():
        train_path = data_dir / 'tokenized' / 'train' / 'data.jsonl'
    
    val_path = data_dir / 'tokenized' / 'val' / 'data.json'
    if not val_path.exists():
        val_path = data_dir / 'tokenized' / 'val' / 'data.jsonl'
    
    train_loader = MusicDataLoader(
        train_path, 
        batch_size_tokens=args.batch_size_tokens,
        max_seq_length=model_config.get('max_seq_length', 5000),
        shuffle=True
    )
    val_loader = MusicDataLoader(
        val_path,
        batch_size_tokens=args.batch_size_tokens,
        max_seq_length=model_config.get('max_seq_length', 5000),
        shuffle=False
    )
    
    # Estimate number of steps per epoch
    num_steps_per_epoch = len(train_loader)
    total_steps = num_steps_per_epoch * args.max_epochs
    print(f"Steps per epoch: ~{num_steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_lr_schedule(optimizer, total_steps, warmup_steps=args.warmup_steps)
    
    # Create output directory
    output_dir = Path(args.output_dir) / config.get('name', 'model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training metrics
    metrics = TrainingMetrics()
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    training_results = {
        'config': config,
        'training_args': vars(args),
        'model_params': n_params,
        'epochs': []
    }
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, 
            metrics, log_interval=args.log_interval
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        metrics.log_validation(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        epoch_time = metrics.get_epoch_time()
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'gpu_memory_mb': metrics.metrics.get('gpu_memory_mb', [0])[-1] if metrics.metrics.get('gpu_memory_mb') else 0
        }
        training_results['epochs'].append(epoch_results)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'epoch': epoch + 1
            }, output_dir / 'best_model.pt')
        
        # Save training results
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics.get_metrics(), f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return training_results


if __name__ == '__main__':
    main()

