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


def train_one_epoch(model, train_loader, optimizer, scheduler, device, log_interval=100):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    for step, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Clamp token IDs to valid range
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
        
        # Forward pass
        logits, loss = model(input_ids, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_steps += 1
        
        if step % log_interval == 0:
            avg_loss = total_loss / num_steps
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    
    avg_loss = total_loss / num_steps
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
        
        vocab_size = model.vocab_size
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        input_ids = torch.where(input_ids == -1, torch.tensor(0, device=device), input_ids)
        
        logits, loss = model(input_ids, target_ids)
        total_loss += loss.item()
        num_steps += 1
    
    avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
    return avg_loss

