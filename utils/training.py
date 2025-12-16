#!/usr/bin/env python3
"""
Training utilities for music language models.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_lr_schedule(optimizer, num_steps, warmup_steps=0):
    """Create cosine annealing learning rate schedule."""
    if warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    return scheduler


def train_one_epoch(model, train_loader, optimizer, scheduler, device, log_interval=100, gradient_accumulation_steps=1):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0
    skipped_batches = 0
    
    # For MPS, use gradient accumulation to simulate larger batches
    if device.type == 'mps':
        gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    optimizer.zero_grad()
    
    for step, (input_ids, target_ids) in enumerate(train_loader):
        try:
            # Truncate sequences if too long for MPS
            if device.type == 'mps' and input_ids.size(1) > 2000:
                max_len = 2000
                input_ids = input_ids[:, :max_len]
                target_ids = target_ids[:, :max_len]
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Clamp token IDs to valid range
            vocab_size = model.vocab_size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
            input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
            
            # Forward pass
            logits, loss = model(input_ids, target_ids)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights only after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
            num_steps += 1
            
            # Clear device cache frequently to free memory
            if device.type == 'mps' and step % 5 == 0:
                torch.mps.empty_cache()
            elif device.type == 'cuda' and step % 10 == 0:
                # For CUDA, clear cache less frequently but still periodically
                torch.cuda.empty_cache()
            
            if step % log_interval == 0:
                avg_loss = total_loss / num_steps
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "MPS" in str(e):
                print(f"\nWARNING: Out of memory at step {step}")
                print(f"  Error: {e}")
                print(f"  Clearing cache and skipping batch...")
                
                # Clear cache
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Skip this batch and continue
                skipped_batches += 1
                continue
            else:
                raise e
    
    # Handle any remaining gradients
    if num_steps % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Warn if too many batches were skipped
    if skipped_batches > 0:
        total_batches = step + 1
        skip_rate = skipped_batches / total_batches * 100
        print(f"\nWARNING: Skipped {skipped_batches}/{total_batches} batches ({skip_rate:.1f}%) due to OOM")
        if skip_rate > 50:
            print(f"  CRITICAL: More than 50% of batches skipped!")
            print(f"  Recommendations:")
            print(f"    1. Reduce batch size further (currently using device-appropriate size)")
            print(f"    2. Use a smaller model (set USE_XL_MODEL=False to use 'large' instead)")
            print(f"    3. Reduce max_seq_length further")
            print(f"    4. Restart kernel to clear memory and try again")
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_steps = 0
    
    for input_ids, target_ids in val_loader:
        try:
            # Truncate sequences if too long for MPS
            if device.type == 'mps' and input_ids.size(1) > 2000:
                max_len = 2000
                input_ids = input_ids[:, :max_len]
                target_ids = target_ids[:, :max_len]
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            vocab_size = model.vocab_size
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
            input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
            
            logits, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            num_steps += 1
            
            # Clear MPS cache periodically
            if device.type == 'mps' and num_steps % 5 == 0:
                torch.mps.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "MPS" in str(e):
                print(f"WARNING: OOM during evaluation, skipping batch")
                if device.type == 'mps':
                    torch.mps.empty_cache()
                continue
            else:
                raise e
    
    avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
    return avg_loss

