#!/usr/bin/env python3
"""
Sample Generation: Train best model and generate music samples.
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import GPT as MusicTransformer
from utils.data_loader import MusicDataLoader, load_tokenizer
from utils.training import train_one_epoch, evaluate, get_lr_schedule
from torch.optim import AdamW


# Model configurations (same as scaling study)
TRANSFORMER_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 2,
        'd_ff': 512,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 1e6
    },
    'small': {
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'd_ff': 1024,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 5e6
    },
    'medium': {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 2048,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 20e6
    },
    'large': {
        'd_model': 768,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 3072,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 50e6
    },
    'xl': {
        'd_model': 1024,
        'n_layers': 12,
        'n_heads': 12,
        'd_ff': 4096,
        'max_seq_length': 5000,
        'dropout': 0.1,
        'target_params': 100e6
    }
}


def train_best_model(
    tokenizer,
    train_path,
    val_path,
    test_path,
    output_dir,
    device,
    model_name='xl',
    num_epochs=3,
    learning_rate=3e-4,
    batch_size_tokens=50000,
    log_interval=100
):
    """Train the best (largest) model for sample generation."""
    print("="*60)
    print("TRAINING BEST MODEL FOR SAMPLE GENERATION")
    print("="*60)
    
    # Get best config
    if model_name not in TRANSFORMER_CONFIGS:
        model_name = max(TRANSFORMER_CONFIGS.keys(), 
                        key=lambda k: TRANSFORMER_CONFIGS[k]['target_params'])
    best_config = TRANSFORMER_CONFIGS[model_name]
    
    print(f"Training {model_name.upper()} model for sample generation...")
    print(f"Config: {best_config}")
    
    # Initialize best model
    best_model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=best_config['d_model'],
        n_layers=best_config['n_layers'],
        n_heads=best_config['n_heads'],
        d_ff=best_config['d_ff'],
        max_seq_length=best_config['max_seq_length'],
        dropout=best_config['dropout']
    ).to(device)
    
    num_params = best_model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create data loaders
    train_loader = MusicDataLoader(
        train_path,
        batch_size_tokens=batch_size_tokens,
        max_seq_length=best_config['max_seq_length'],
        shuffle=True
    )
    
    val_loader = MusicDataLoader(
        val_path,
        batch_size_tokens=batch_size_tokens,
        max_seq_length=best_config['max_seq_length'],
        shuffle=False
    )
    
    # Setup optimizer and scheduler
    estimated_steps = len(train_loader) if hasattr(train_loader, '__len__') else 1000
    optimizer = AdamW(best_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_lr_schedule(optimizer, estimated_steps, warmup_steps=0)
    
    # Train for multiple epochs
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss = train_one_epoch(
            best_model, train_loader, optimizer, scheduler, device,
            log_interval=log_interval
        )
        
        val_loss = evaluate(best_model, val_loader, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
    
    # Final evaluation on test set
    test_loader = MusicDataLoader(
        test_path,
        batch_size_tokens=batch_size_tokens,
        max_seq_length=best_config['max_seq_length'],
        shuffle=False
    )
    
    test_loss = evaluate(best_model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # Save best model
    best_model_path = output_dir / "best_model.pt"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_config': best_config,
        'vocab_size': tokenizer.vocab_size,
        'num_parameters': num_params,
        'test_loss': test_loss,
    }, best_model_path)
    
    print(f"\nBest model saved to: {best_model_path}")
    
    return best_model, test_loss


def generate_samples(
    model,
    tokenizer,
    output_dir,
    device,
    num_samples=10,
    max_new_tokens=500,
    temperature=1.0,
    top_k=50
):
    """Generate music samples from the trained model."""
    print("\n" + "="*60)
    print("GENERATING MUSIC SAMPLES")
    print("="*60)
    
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    generated_samples = []
    
    print(f"Generating {num_samples} samples...")
    
    for i in range(num_samples):
        # Unconditional generation: start with a random token or special token
        start_token_id = tokenizer.token_to_id.get('<START>', 2)  # Use START token if available
        if start_token_id not in tokenizer.token_to_id.values():
            # If no START token, use a common note token
            start_token_id = tokenizer.token_to_id.get('C', 5)  # Start with C note
        
        start_tokens = torch.tensor([[start_token_id]], device=device)
        
        with torch.no_grad():
            generated = model.generate(
                start_tokens, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_k=top_k  # Top-k sampling for diversity
            )
        
        # Decode generated tokens
        generated_tokens = generated[0].cpu().tolist()
        generated_abc = tokenizer.decode(generated_tokens)
        
        # Save sample
        sample_path = samples_dir / f"sample_{i+1}.abc"
        with open(sample_path, 'w') as f:
            f.write(generated_abc)
        
        generated_samples.append({
            'sample_id': i+1,
            'tokens': generated_tokens,
            'abc': generated_abc,
            'length': len(generated_tokens)
        })
        
        print(f"Sample {i+1}: {len(generated_tokens)} tokens")
        print(f"  Saved to: {sample_path}")
        print(f"  Preview: {generated_abc[:100]}...")
    
    # Save samples metadata
    with open(samples_dir / "samples_metadata.json", 'w') as f:
        json.dump(generated_samples, f, indent=2)
    
    print(f"\n✓ Generated {num_samples} samples")
    print(f"Samples saved to: {samples_dir}")
    
    return generated_samples


def evaluate_samples(generated_samples, test_loss):
    """Evaluate generated samples."""
    print("\n" + "="*60)
    print("SAMPLE EVALUATION")
    print("="*60)
    
    # Try to convert samples to MIDI for validation
    valid_samples = 0
    for i, sample in enumerate(generated_samples):
        try:
            # Try to parse ABC notation (basic validation)
            abc_str = sample['abc']
            # Check if it has basic ABC structure
            if 'X:' in abc_str or len(abc_str.strip()) > 10:
                valid_samples += 1
        except:
            pass
    
    num_samples = len(generated_samples)
    print(f"Valid samples (basic syntax check): {valid_samples}/{num_samples} ({valid_samples/num_samples*100:.1f}%)")
    print(f"Average sample length: {np.mean([s['length'] for s in generated_samples]):.1f} tokens")
    print(f"Test set perplexity: {np.exp(test_loss):.2f}")
    
    # Save evaluation results
    evaluation_results = {
        'test_loss': float(test_loss),
        'test_perplexity': float(np.exp(test_loss)),
        'num_samples': num_samples,
        'valid_samples': valid_samples,
        'valid_percentage': float(valid_samples/num_samples*100),
        'average_sample_length': float(np.mean([s['length'] for s in generated_samples]))
    }
    
    return evaluation_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train best model and generate samples")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--output-dir", type=str, default="outputs/samples",
                       help="Output directory for results")
    parser.add_argument("--model-name", type=str, default="xl",
                       choices=['tiny', 'small', 'medium', 'large', 'xl'],
                       help="Model size to train")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to generate")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                       help="Maximum tokens to generate per sample")
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
    test_path = data_dir / "tokenized" / "test" / "data.json"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train best model
    best_model, test_loss = train_best_model(
        tokenizer, train_path, val_path, test_path, output_dir, device,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size_tokens=args.batch_size_tokens
    )
    
    # Generate samples
    generated_samples = generate_samples(
        best_model, tokenizer, output_dir, device,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    # Evaluate samples
    evaluation_results = evaluate_samples(generated_samples, test_loss)
    
    # Save evaluation results
    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {eval_path}")

