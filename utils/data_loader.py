#!/usr/bin/env python3
"""
Data loading utilities for tokenized music sequences.
"""

import json
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


def load_tokenizer(tokenizer_path: Path):
    """Load tokenizer from pickle file."""
    import sys
    import importlib.util
    from pathlib import Path as PathLib
    
    # Add parent directory to path to import MusicTokenizer
    project_root = PathLib(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Try to import from the new tokenizer.py file first
    tokenizer_module_path = project_root / "data-collection-and-preprocessing" / "tokenizer.py"
    
    if tokenizer_module_path.exists():
        spec = importlib.util.spec_from_file_location("tokenizer", tokenizer_module_path)
        tokenizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizer_module)
        MusicTokenizer = tokenizer_module.MusicTokenizer
    else:
        # Fallback: try to load from data_preprocessing.py (old location)
        preprocessing_path = project_root / "data-collection-and-preprocessing" / "data_preprocessing.py"
        
        if preprocessing_path.exists():
            spec = importlib.util.spec_from_file_location("data_preprocessing", preprocessing_path)
            data_preprocessing = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_preprocessing)
            MusicTokenizer = data_preprocessing.MusicTokenizer
        else:
            # Last resort: try direct import
            try:
                from data_collection_and_preprocessing.tokenizer import MusicTokenizer
            except ImportError:
                # Create a minimal tokenizer class
                class MusicTokenizer:
                    def __init__(self):
                        self.vocab = {}
                        self.token_to_id = {}
                        self.id_to_token = {}
                        self.vocab_size = 0
                        self.special_tokens = {}
    
    # Load tokenizer using the class method if available, otherwise reconstruct manually
    try:
        tokenizer = MusicTokenizer.load(tokenizer_path)
    except (AttributeError, TypeError):
        # Fallback: reconstruct manually
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = MusicTokenizer()
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.vocab_size = tokenizer_data['vocab_size']
        tokenizer.special_tokens = tokenizer_data['special_tokens']
    
    return tokenizer


class MusicDataset(Dataset):
    """Dataset for tokenized music sequences."""
    
    def __init__(self, data_path: Path, max_seq_length: int = 5000):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON or JSONL file with tokenized sequences
            max_seq_length: Maximum sequence length (will truncate if longer)
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.sequences = []
        
        # Load sequences from JSON or JSONL file
        print(f"Loading sequences from {data_path}...")
        with open(data_path, 'r') as f:
            # Try to load as JSON array first
            content = f.read()
            try:
                data = json.loads(content)
                # If it's a list, it's a JSON array
                if isinstance(data, list):
                    for item in data:
                        # Handle both 'token_ids' and 'tokens' field names
                        token_ids = item.get('token_ids') or item.get('tokens', [])
                        # Truncate if too long
                        if len(token_ids) > max_seq_length:
                            token_ids = token_ids[:max_seq_length]
                        if len(token_ids) > 0:  # Only add non-empty sequences
                            self.sequences.append(token_ids)
                else:
                    # Single object
                    token_ids = data.get('token_ids') or data.get('tokens', [])
                    if len(token_ids) > max_seq_length:
                        token_ids = token_ids[:max_seq_length]
                    if len(token_ids) > 0:
                        self.sequences.append(token_ids)
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                f.seek(0)
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        token_ids = data.get('token_ids') or data.get('tokens', [])
                        if len(token_ids) > max_seq_length:
                            token_ids = token_ids[:max_seq_length]
                        if len(token_ids) > 0:
                            self.sequences.append(token_ids)
        
        print(f"Loaded {len(self.sequences)} sequences")
        if len(self.sequences) > 0:
            print(f"Average sequence length: {sum(len(s) for s in self.sequences) / len(self.sequences):.1f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Convert to tensor
        tokens = torch.tensor(sequence, dtype=torch.long)
        # Input is all tokens except last, target is all tokens except first
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


class MusicDataLoader:
    """Data loader that batches by tokens (not sequences)."""
    
    def __init__(self, data_path: Path, batch_size_tokens: int, 
                 max_seq_length: int = 5000, shuffle: bool = True,
                 num_workers: int = 0):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to JSONL file with tokenized sequences
            batch_size_tokens: Target number of tokens per batch
            max_seq_length: Maximum sequence length
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
        """
        self.data_path = data_path
        self.batch_size_tokens = batch_size_tokens
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        
        # Create dataset
        self.dataset = MusicDataset(data_path, max_seq_length)
        
        # Create data loader with custom collate function
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # We'll handle batching manually
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to create token-based batches."""
        # batch is a list of (input_ids, target_ids) tuples
        # We'll return them as-is and handle batching in __iter__
        return batch[0]  # Return single item since batch_size=1
    
    def __iter__(self):
        """Create batches based on token count."""
        batch_inputs = []
        batch_targets = []
        current_batch_tokens = 0
        
        for input_ids, target_ids in self.dataloader:
            seq_len = input_ids.size(0)
            
            # If adding this sequence would exceed batch size, yield current batch
            if current_batch_tokens + seq_len > self.batch_size_tokens and len(batch_inputs) > 0:
                # Pad sequences to same length
                max_len = max(seq.size(0) for seq in batch_inputs)
                padded_inputs = []
                padded_targets = []
                
                for inp, tgt in zip(batch_inputs, batch_targets):
                    pad_len = max_len - inp.size(0)
                    if pad_len > 0:
                        # Pad with -1 (will be ignored in loss)
                        inp = torch.cat([inp, torch.full((pad_len,), -1, dtype=inp.dtype)])
                        tgt = torch.cat([tgt, torch.full((pad_len,), -1, dtype=tgt.dtype)])
                    padded_inputs.append(inp)
                    padded_targets.append(tgt)
                
                yield torch.stack(padded_inputs), torch.stack(padded_targets)
                
                # Reset batch
                batch_inputs = []
                batch_targets = []
                current_batch_tokens = 0
            
            # Add to current batch
            batch_inputs.append(input_ids)
            batch_targets.append(target_ids)
            current_batch_tokens += seq_len
        
        # Yield remaining batch
        if len(batch_inputs) > 0:
            max_len = max(seq.size(0) for seq in batch_inputs)
            padded_inputs = []
            padded_targets = []
            
            for inp, tgt in zip(batch_inputs, batch_targets):
                pad_len = max_len - inp.size(0)
                if pad_len > 0:
                    inp = torch.cat([inp, torch.full((pad_len,), -1, dtype=inp.dtype)])
                    tgt = torch.cat([tgt, torch.full((pad_len,), -1, dtype=tgt.dtype)])
                padded_inputs.append(inp)
                padded_targets.append(tgt)
            
            yield torch.stack(padded_inputs), torch.stack(padded_targets)
    
    def __len__(self):
        """Estimate number of batches (approximate)."""
        total_tokens = sum(len(seq) for seq in self.dataset.sequences)
        return max(1, total_tokens // self.batch_size_tokens)

