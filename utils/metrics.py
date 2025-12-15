#!/usr/bin/env python3
"""
Training metrics tracking.
"""

import time
import torch
from collections import defaultdict
from typing import Dict, List


class TrainingMetrics:
    """Track training metrics over time."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        self.step_times = []
        
    def start_epoch(self):
        """Start timing an epoch."""
        self.start_time = time.time()
        
    def log_step(self, step: int, loss: float, lr: float = None, 
                 gpu_memory_mb: float = None):
        """Log metrics for a training step."""
        self.metrics['step'].append(step)
        self.metrics['train_loss'].append(loss)
        if lr is not None:
            self.metrics['learning_rate'].append(lr)
        if gpu_memory_mb is not None:
            self.metrics['gpu_memory_mb'].append(gpu_memory_mb)
        
        # Track step time
        if len(self.step_times) > 0:
            step_time = time.time() - self.step_times[-1]
            self.metrics['step_time'].append(step_time)
        self.step_times.append(time.time())
    
    def log_validation(self, val_loss: float):
        """Log validation loss."""
        self.metrics['val_loss'].append(val_loss)
    
    def get_epoch_time(self) -> float:
        """Get elapsed time for current epoch."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_metrics(self) -> Dict:
        """Get all metrics as dictionary."""
        return dict(self.metrics)
    
    def get_latest(self) -> Dict:
        """Get latest values for all metrics."""
        latest = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                latest[key] = values[-1]
        return latest
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.start_time = None
        self.step_times = []


