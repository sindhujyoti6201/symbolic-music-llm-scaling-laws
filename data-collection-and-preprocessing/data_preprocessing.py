#!/usr/bin/env python3
"""
Data Preprocessing Script for Symbolic Music LLM Project
Converts MIDI files to ABC notation, tokenizes, cleans, and splits data.
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Aggressively suppress all warnings, especially from music21
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress music21 specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*TranslateWarning.*')
warnings.filterwarnings('ignore', message='.*Unable to decode.*')
warnings.filterwarnings('ignore', message='.*Unable to determine.*')

import music21
# Suppress music21's own warning system
music21.environment.UserSettings()['warnings'] = 0

import numpy as np
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
LMD_DIR = DATA_DIR / "lmd_matched"
MATCH_SCORES = DATA_DIR / "match_scores.json"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "abc").mkdir(exist_ok=True)
(OUTPUT_DIR / "tokenized").mkdir(exist_ok=True)

# Preprocessing parameters
MIN_SEQUENCE_LENGTH = 50  # Minimum tokens per sequence
MAX_SEQUENCE_LENGTH = 5000  # Maximum tokens per sequence
TRAIN_SPLIT = 0.98
VAL_SPLIT = 0.01
TEST_SPLIT = 0.01
MIN_TRAIN_TOKENS = 100_000_000  # 100M tokens minimum


class MIDIToABCConverter:
    """Convert MIDI files to ABC notation using music21."""
    
    def __init__(self):
        self.conversion_stats = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
    
    def convert_midi_to_abc(self, midi_path: Path) -> Optional[str]:
        """
        Convert a MIDI file to ABC notation.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            ABC notation string or None if conversion fails
        """
        try:
            # Suppress all warnings and stderr output for this conversion
            import warnings
            import io
            import contextlib
            
            # Create a null device to discard stderr
            null_stderr = io.StringIO()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stderr(null_stderr):
                    # Parse MIDI file (suppress all output)
                    score = music21.converter.parse(str(midi_path))
                    
                    # Convert to ABC notation using music21's ABC format
                    # music21.write() returns a file path, so we need to read it
                    temp_path = score.write('abc')
            
            # Read the ABC file content
            with open(temp_path, 'r') as f:
                abc_str = f.read()
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Clean up ABC string
            abc_str = self._clean_abc(abc_str)
            
            if len(abc_str.strip()) > 0:
                self.conversion_stats['success'] += 1
                return abc_str
            else:
                self.conversion_stats['failed'] += 1
                return None
                
        except Exception as e:
            self.conversion_stats['failed'] += 1
            self.conversion_stats['errors'].append({
                'file': str(midi_path),
                'error': str(e)
            })
            return None
    
    def _clean_abc(self, abc_str: str) -> str:
        """Clean and normalize ABC notation string."""
        lines = abc_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and some metadata
            if line and not line.startswith('%'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class MusicTokenizer:
    """
    Tokenizer for ABC notation.
    Uses note-level tokenization with music-aware tokens.
    """
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<SEP>': 4,  # Separator between measures/sections
        }
    
    def build_vocab(self, abc_strings: List[str], min_freq: int = 2):
        """
        Build vocabulary from ABC strings.
        
        Args:
            abc_strings: List of ABC notation strings
            min_freq: Minimum frequency for a token to be included
        """
        print("Building vocabulary...")
        
        # Tokenize all strings and count frequencies
        token_counter = Counter()
        
        for abc_str in tqdm(abc_strings, desc="Tokenizing for vocab"):
            tokens = self._tokenize_abc(abc_str)
            token_counter.update(tokens)
        
        # Add special tokens
        vocab = dict(self.special_tokens)
        current_id = len(self.special_tokens)
        
        # Add tokens that meet minimum frequency
        for token, count in token_counter.items():
            if count >= min_freq:
                vocab[token] = current_id
                current_id += 1
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Regular tokens: {self.vocab_size - len(self.special_tokens)}")
    
    def _tokenize_abc(self, abc_str: str) -> List[str]:
        """
        Tokenize ABC notation string into music-aware tokens.
        
        Tokenization strategy:
        - Notes: C, D, E, F, G, A, B (with accidentals: ^, _)
        - Octaves: , (lower) and ' (higher)
        - Durations: numbers (1, 2, 4, 8, etc.)
        - Rests: z
        - Bar lines: |
        - Other ABC symbols as separate tokens
        """
        tokens = []
        i = 0
        
        while i < len(abc_str):
            char = abc_str[i]
            
            # Skip whitespace (we'll use it as separator)
            if char.isspace():
                i += 1
                continue
            
            # Bar lines
            if char == '|':
                tokens.append('|')
                i += 1
                continue
            
            # Notes (A-G, a-g)
            if char.upper() in 'ABCDEFG':
                note_token = char.upper()
                i += 1
                
                # Check for accidental (^ or _)
                if i < len(abc_str) and abc_str[i] in '^_':
                    note_token += abc_str[i]
                    i += 1
                
                # Check for octave markers (,' or ')
                while i < len(abc_str) and abc_str[i] in ",'":
                    note_token += abc_str[i]
                    i += 1
                
                tokens.append(note_token)
                continue
            
            # Durations (numbers)
            if char.isdigit():
                duration = char
                i += 1
                while i < len(abc_str) and abc_str[i].isdigit():
                    duration += abc_str[i]
                    i += 1
                tokens.append(f"DUR:{duration}")
                continue
            
            # Rests
            if char == 'z':
                tokens.append('z')
                i += 1
                continue
            
            # Other characters as individual tokens
            tokens.append(char)
            i += 1
        
        return tokens
    
    def encode(self, abc_str: str) -> List[int]:
        """Encode ABC string to token IDs."""
        tokens = self._tokenize_abc(abc_str)
        token_ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id['<UNK>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to ABC string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        
        # Reconstruct ABC string (simplified)
        return ' '.join(tokens)
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = data['id_to_token']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.special_tokens = data['special_tokens']
        
        return tokenizer


def _convert_single_midi(args: Tuple[Path, Path]) -> Optional[Tuple[Path, str]]:
    """
    Convert a single MIDI file to ABC (for parallel processing).
    
    Args:
        args: Tuple of (midi_path, output_dir)
        
    Returns:
        (midi_path, abc_string) tuple or None if conversion fails
    """
    midi_file, output_dir = args
    
    # Create a new converter instance for this process
    converter = MIDIToABCConverter()
    abc_str = converter.convert_midi_to_abc(midi_file)
    
    if abc_str:
        # Save ABC file
        abc_path = output_dir / "abc" / f"{midi_file.stem}.abc"
        abc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abc_path, 'w') as f:
            f.write(abc_str)
        
        return (midi_file, abc_str)
    
    return None


def process_midi_files(
    midi_files: List[Path],
    converter: MIDIToABCConverter,
    output_dir: Path,
    max_files: Optional[int] = None,
    num_workers: Optional[int] = None
) -> List[Tuple[Path, str]]:
    """
    Process MIDI files and convert to ABC notation (parallelized).
    
    Args:
        midi_files: List of MIDI file paths
        converter: MIDIToABCConverter instance (not used in parallel mode)
        output_dir: Output directory for ABC files
        max_files: Maximum number of files to process
        num_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        List of (midi_path, abc_string) tuples
    """
    print(f"\nProcessing {len(midi_files)} MIDI files...")
    
    if max_files:
        midi_files = midi_files[:max_files]
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    args_list = [(midi_file, output_dir) for midi_file in midi_files]
    
    # Process files in parallel
    abc_data = []
    conversion_stats = {'success': 0, 'failed': 0}
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_convert_single_midi, args_list),
            total=len(args_list),
            desc="Converting MIDI to ABC"
        ))
    
    # Collect results
    for result in results:
        if result is not None:
            abc_data.append(result)
            conversion_stats['success'] += 1
        else:
            conversion_stats['failed'] += 1
    
    print(f"\nConversion statistics:")
    print(f"  Successful: {conversion_stats['success']}")
    print(f"  Failed: {conversion_stats['failed']}")
    
    return abc_data


def filter_sequences(
    abc_data: List[Tuple[Path, str]],
    tokenizer: MusicTokenizer,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH
) -> List[Tuple[Path, str, List[int]]]:
    """
    Filter sequences by length and tokenize.
    
    Returns:
        List of (midi_path, abc_string, token_ids) tuples
    """
    print(f"\nFiltering sequences (length: {min_length}-{max_length} tokens)...")
    
    filtered_data = []
    stats = {
        'too_short': 0,
        'too_long': 0,
        'valid': 0
    }
    
    for midi_path, abc_str in tqdm(abc_data, desc="Filtering"):
        token_ids = tokenizer.encode(abc_str)
        seq_length = len(token_ids)
        
        if seq_length < min_length:
            stats['too_short'] += 1
            continue
        elif seq_length > max_length:
            stats['too_long'] += 1
            continue
        else:
            stats['valid'] += 1
            filtered_data.append((midi_path, abc_str, token_ids))
    
    print(f"\nFiltering statistics:")
    print(f"  Too short (<{min_length}): {stats['too_short']}")
    print(f"  Too long (>{max_length}): {stats['too_long']}")
    print(f"  Valid: {stats['valid']}")
    
    return filtered_data


def create_splits(
    data: List[Tuple[Path, str, List[int]]],
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT
) -> Tuple[List, List, List]:
    """Create train/validation/test splits."""
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = [data[i] for i in indices]
    
    # Calculate split indices
    n_total = len(data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data


def save_splits(
    train_data: List,
    val_data: List,
    test_data: List,
    output_dir: Path
):
    """Save train/val/test splits to disk."""
    print("\nSaving splits...")
    
    def save_split(data, split_name):
        split_dir = output_dir / "tokenized" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy loading
        json_data = []
        for midi_path, abc_str, token_ids in data:
            json_data.append({
                'midi_path': str(midi_path),
                'abc': abc_str,
                'tokens': token_ids,
                'length': len(token_ids)
            })
        
        with open(split_dir / "data.json", 'w') as f:
            json.dump(json_data, f, indent=2)
    
    save_split(train_data, "train")
    save_split(val_data, "val")
    save_split(test_data, "test")
    
    print(f"  Train: {len(train_data)} sequences")
    print(f"  Val: {len(val_data)} sequences")
    print(f"  Test: {len(test_data)} sequences")


def generate_statistics(
    train_data: List,
    val_data: List,
    test_data: List,
    tokenizer: MusicTokenizer,
    output_dir: Path
):
    """Generate and save preprocessing statistics."""
    print("\nGenerating statistics...")
    
    def count_tokens(data):
        return sum(len(tokens) for _, _, tokens in data)
    
    train_tokens = count_tokens(train_data)
    val_tokens = count_tokens(val_data)
    test_tokens = count_tokens(test_data)
    total_tokens = train_tokens + val_tokens + test_tokens
    
    stats = {
        'dataset_info': {
            'total_sequences': len(train_data) + len(val_data) + len(test_data),
            'train_sequences': len(train_data),
            'val_sequences': len(val_data),
            'test_sequences': len(test_data),
        },
        'token_counts': {
            'total_tokens': total_tokens,
            'train_tokens': train_tokens,
            'val_tokens': val_tokens,
            'test_tokens': test_tokens,
        },
        'tokenization': {
            'vocab_size': tokenizer.vocab_size,
            'tokenization_scheme': 'Note-level with music-aware tokens',
            'special_tokens': list(tokenizer.special_tokens.keys()),
        },
        'filtering_criteria': {
            'min_sequence_length': MIN_SEQUENCE_LENGTH,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'min_train_tokens_required': MIN_TRAIN_TOKENS,
            'min_train_tokens_achieved': train_tokens,
            'meets_requirement': train_tokens >= MIN_TRAIN_TOKENS,
        },
        'splits': {
            'train': TRAIN_SPLIT,
            'val': VAL_SPLIT,
            'test': TEST_SPLIT,
        }
    }
    
    # Save statistics
    with open(output_dir / "preprocessing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING STATISTICS")
    print("=" * 60)
    print(f"\nDataset:")
    print(f"  Total sequences: {stats['dataset_info']['total_sequences']:,}")
    print(f"  Train: {stats['dataset_info']['train_sequences']:,}")
    print(f"  Val: {stats['dataset_info']['val_sequences']:,}")
    print(f"  Test: {stats['dataset_info']['test_sequences']:,}")
    
    print(f"\nTokens:")
    print(f"  Total: {total_tokens:,}")
    print(f"  Train: {train_tokens:,} ({train_tokens/1e6:.1f}M)")
    print(f"  Val: {val_tokens:,}")
    print(f"  Test: {test_tokens:,}")
    
    print(f"\nTokenization:")
    print(f"  Scheme: {stats['tokenization']['tokenization_scheme']}")
    print(f"  Vocabulary size: {stats['tokenization']['vocab_size']:,}")
    
    print(f"\nFiltering:")
    print(f"  Sequence length: {MIN_SEQUENCE_LENGTH}-{MAX_SEQUENCE_LENGTH} tokens")
    print(f"  Train tokens requirement: {MIN_TRAIN_TOKENS:,} ({MIN_TRAIN_TOKENS/1e6:.0f}M)")
    print(f"  Train tokens achieved: {train_tokens:,} ({train_tokens/1e6:.1f}M)")
    print(f"  Meets requirement: {'✓ YES' if train_tokens >= MIN_TRAIN_TOKENS else '✗ NO'}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess LMD dataset')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of MIDI files to process (for testing)')
    parser.add_argument('--skip-conversion', action='store_true',
                        help='Skip MIDI to ABC conversion (use existing ABC files)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count - 1, current: {max(1, cpu_count() - 1)})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Get MIDI files
    print("\nStep 1: Finding MIDI files...")
    if not LMD_DIR.exists():
        print(f"ERROR: {LMD_DIR} not found!")
        print("Please run data_collection.py first to download the dataset.")
        return 1
    
    midi_files = list(LMD_DIR.rglob("*.mid"))
    print(f"Found {len(midi_files):,} MIDI files")
    
    if len(midi_files) == 0:
        print("No MIDI files found!")
        return 1
    
    # Step 2: Convert MIDI to ABC
    converter = MIDIToABCConverter()
    
    if not args.skip_conversion:
        abc_data = process_midi_files(
            midi_files,
            converter,
            OUTPUT_DIR,
            max_files=args.max_files,
            num_workers=args.num_workers
        )
    else:
        # Load existing ABC files
        abc_files = list((OUTPUT_DIR / "abc").glob("*.abc"))
        abc_data = []
        for abc_file in abc_files:
            with open(abc_file, 'r') as f:
                abc_str = f.read()
            abc_data.append((abc_file, abc_str))
        print(f"Loaded {len(abc_data)} existing ABC files")
    
    if len(abc_data) == 0:
        print("No ABC data available!")
        return 1
    
    # Step 3: Build vocabulary
    print("\nStep 3: Building vocabulary...")
    tokenizer = MusicTokenizer()
    abc_strings = [abc_str for _, abc_str in abc_data]
    tokenizer.build_vocab(abc_strings, min_freq=2)
    
    # Save tokenizer
    tokenizer.save(OUTPUT_DIR / "tokenizer.pkl")
    print(f"Tokenizer saved to {OUTPUT_DIR / 'tokenizer.pkl'}")
    
    # Step 4: Filter sequences
    print("\nStep 4: Filtering sequences...")
    filtered_data = filter_sequences(abc_data, tokenizer)
    
    if len(filtered_data) == 0:
        print("No valid sequences after filtering!")
        return 1
    
    # Step 5: Create splits
    print("\nStep 5: Creating train/val/test splits...")
    train_data, val_data, test_data = create_splits(filtered_data)
    
    # Step 6: Save splits
    save_splits(train_data, val_data, test_data, OUTPUT_DIR)
    
    # Step 7: Generate statistics
    generate_statistics(train_data, val_data, test_data, tokenizer, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - ABC files: {OUTPUT_DIR / 'abc'}")
    print(f"  - Tokenized data: {OUTPUT_DIR / 'tokenized'}")
    print(f"  - Tokenizer: {OUTPUT_DIR / 'tokenizer.pkl'}")
    print(f"  - Statistics: {OUTPUT_DIR / 'preprocessing_stats.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

