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
        Uses manual conversion since music21's ABC converter may not be registered.
        
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
            
            # Convert to ABC manually (since ABC converter is not registered)
            abc_str = self._score_to_abc_manual(score)
            
            if abc_str:
                # Clean up ABC string
                abc_str = self._clean_abc(abc_str)
                
                if len(abc_str.strip()) > 0:
                    self.conversion_stats['success'] += 1
                    return abc_str
            
            self.conversion_stats['failed'] += 1
            print(f"WARNING: Empty ABC string for {midi_path.name}")
            return None
                
        except Exception as e:
            self.conversion_stats['failed'] += 1
            self.conversion_stats['errors'].append({
                'file': str(midi_path),
                'error': str(e),
                'error_type': type(e).__name__
            })
            print(f"ERROR in convert_midi_to_abc for {midi_path.name}: {type(e).__name__}: {e}")
            return None
    
    def _score_to_abc_manual(self, score) -> str:
        """
        Manually convert a music21 score to ABC notation.
        This extracts notes and constructs ABC notation directly.
        """
        try:
            abc_lines = []
            
            # ABC header
            abc_lines.append("X:1")
            abc_lines.append("T:Untitled")
            
            # Get time signature
            time_sig = score.flat.getElementsByClass('TimeSignature')
            if time_sig:
                ts = time_sig[0]
                abc_lines.append(f"M:{ts.numerator}/{ts.denominator}")
            else:
                abc_lines.append("M:4/4")  # Default
            
            # Get key signature
            key_sig = score.flat.getElementsByClass('KeySignature')
            if key_sig:
                key = key_sig[0]
                key_name = key.tonic.name if hasattr(key, 'tonic') else 'C'
                mode = 'maj' if key.mode == 'major' else 'm'
                abc_lines.append(f"K:{key_name}{mode}")
            else:
                abc_lines.append("K:C")  # Default
            
            abc_lines.append("L:1/8")   # Default note length
            
            # Process each part
            notes_sequence = []
            
            for part in score.parts:
                part_notes = []
                
                # Get all notes and rests from this part
                for element in part.flat.notesAndRests:
                    if element.isNote:
                        # Single note
                        note = element
                        abc_note = self._note_to_abc(note)
                        if abc_note:
                            part_notes.append(abc_note)
                    elif element.isChord:
                        # Chord - take the first note
                        note = element.notes[0]
                        abc_note = self._note_to_abc(note)
                        if abc_note:
                            part_notes.append(abc_note)
                    elif element.isRest:
                        # Rest
                        dur = self._duration_to_abc(element.duration.quarterLength)
                        part_notes.append(f"z{dur}")
                
                if part_notes:
                    notes_sequence.extend(part_notes)
            
            # Format notes with proper spacing and bar lines
            if notes_sequence:
                abc_body = []
                measure_count = 0
                notes_per_measure = 8  # Default, will adjust based on time signature
                
                # Try to determine notes per measure from time signature
                if time_sig:
                    ts = time_sig[0]
                    # Approximate: 4/4 time = 8 eighth notes per measure
                    notes_per_measure = int(ts.numerator * 2)  # Assuming 1/8 notes
                
                for i, note in enumerate(notes_sequence):
                    # Add space between notes for readability
                    if i > 0:
                        abc_body.append(" ")
                    
                    # Add bar line at measure boundaries
                    if i > 0 and measure_count > 0 and measure_count % notes_per_measure == 0:
                        abc_body.append("| ")
                    
                    abc_body.append(note)
                    measure_count += 1
                
                # Join notes into body line (ABC allows long lines, but we'll add some line breaks)
                body_str = "".join(abc_body)
                
                # Add line breaks every ~80 characters for readability (optional)
                if len(body_str) > 80:
                    # Try to break at bar lines
                    parts = body_str.split("|")
                    formatted_parts = []
                    for part in parts:
                        if len(part) > 80:
                            # Break long measures
                            words = part.split()
                            line = []
                            for word in words:
                                if len(" ".join(line + [word])) > 80 and line:
                                    formatted_parts.append(" ".join(line))
                                    line = [word]
                                else:
                                    line.append(word)
                            if line:
                                formatted_parts.append(" ".join(line))
                        else:
                            formatted_parts.append(part)
                    body_str = "|".join(formatted_parts)
                
                abc_lines.append(body_str)
            
            return "\n".join(abc_lines) if abc_lines else ""
            
        except Exception as e:
            print(f"Error in _score_to_abc_manual: {e}")
            return ""
    
    def _note_to_abc(self, note) -> str:
        """Convert a music21 note to ABC notation."""
        try:
            # Get note name (C, D, E, F, G, A, B)
            note_name = note.pitch.name[0]  # Just the letter
            
            # Handle accidentals
            if note.pitch.accidental:
                if note.pitch.accidental.alter == 1:
                    note_name = "^" + note_name  # Sharp
                elif note.pitch.accidental.alter == -1:
                    note_name = "_" + note_name  # Flat
            
            # Handle octave
            octave = note.pitch.octave
            if octave < 4:
                # Lower octaves: use lowercase
                note_name = note_name.lower() * (4 - octave)
            elif octave > 4:
                # Higher octaves: use apostrophes
                note_name = note_name + "'" * (octave - 4)
            
            # Handle duration
            dur = self._duration_to_abc(note.duration.quarterLength)
            
            return note_name + dur
            
        except Exception:
            return ""
    
    def _duration_to_abc(self, quarter_length: float) -> str:
        """
        Convert duration in quarter notes to ABC notation.
        With L:1/8 (default note length = 1/8):
        - 1/8 note = "" (default, no number)
        - 1/16 note = "/" (half of default)
        - 1/4 note = "2" (2 * 1/8)
        - 1/2 note = "4" (4 * 1/8)
        - whole note = "8" (8 * 1/8)
        """
        # Convert quarter_length to number of 1/8 notes
        eighth_notes = quarter_length * 2
        
        # Round to nearest reasonable value
        eighth_notes = round(eighth_notes * 8) / 8  # Round to nearest 1/8
        
        if eighth_notes <= 0:
            return ""
        elif eighth_notes == 0.5:  # 1/16
            return "/"
        elif eighth_notes == 1.0:  # 1/8 (default)
            return ""
        elif eighth_notes == 2.0:  # 1/4
            return "2"
        elif eighth_notes == 3.0:  # dotted 1/4
            return "3"
        elif eighth_notes == 4.0:  # 1/2
            return "4"
        elif eighth_notes == 6.0:  # dotted 1/2
            return "6"
        elif eighth_notes == 8.0:  # whole
            return "8"
        else:
            # For other durations, use the number of 1/8 notes
            dur_int = int(eighth_notes)
            if dur_int > 0 and dur_int <= 16:
                return str(dur_int)
            else:
                # Fallback: use fraction notation
                return f"/{int(1/eighth_notes)}" if eighth_notes < 1 else str(int(eighth_notes))
    
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
    
    try:
        # Create a new converter instance for this process
        converter = MIDIToABCConverter()
        abc_str = converter.convert_midi_to_abc(midi_file)
        
        if abc_str:
            # Save ABC file
            abc_path = output_dir / "abc" / f"{midi_file.stem}.abc"
            abc_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(abc_path, 'w') as f:
                    f.write(abc_str)
                
                # Verify file was actually written
                if abc_path.exists() and abc_path.stat().st_size > 0:
                    return (midi_file, abc_str)
                else:
                    print(f"ERROR: File {abc_path} was not written correctly")
                    return None
            except Exception as write_error:
                print(f"ERROR: Failed to write {abc_path}: {write_error}")
                return None
        
        return None
    except Exception as e:
        # Log the error instead of silently failing
        print(f"ERROR converting {midi_file.name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Force garbage collection to free memory
        import gc
        gc.collect()


def process_midi_files(
    midi_files: List[Path],
    converter: MIDIToABCConverter,
    output_dir: Path,
    max_files: Optional[int] = None,
    num_workers: Optional[int] = None,
    chunk_size: int = 1000
) -> List[Tuple[Path, str]]:
    """
    Process MIDI files and convert to ABC notation (parallelized with chunking).
    
    Args:
        midi_files: List of MIDI file paths
        converter: MIDIToABCConverter instance (not used in parallel mode)
        output_dir: Output directory for ABC files
        max_files: Maximum number of files to process
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Number of files to process in each batch (prevents memory buildup)
        
    Returns:
        List of (midi_path, abc_string) tuples
    """
    print(f"\nProcessing {len(midi_files)} MIDI files...")
    
    if max_files:
        midi_files = midi_files[:max_files]
        print(f"Limited to {max_files} files for processing")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Using {num_workers} parallel workers")
    print(f"Processing in chunks of {chunk_size} files to prevent memory buildup")
    
    # Prepare arguments for parallel processing
    args_list = [(midi_file, output_dir) for midi_file in midi_files]
    
    # Process files in chunks to prevent memory buildup
    abc_data = []
    conversion_stats = {'success': 0, 'failed': 0}
    
    total_chunks = (len(args_list) + chunk_size - 1) // chunk_size
    
    # Checkpoint file for recovery
    checkpoint_file = output_dir / "conversion_checkpoint.json"
    
    # Try to load previous checkpoint
    start_chunk = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            start_chunk = checkpoint.get('last_chunk', 0) + 1
            print(f"\nFound checkpoint: Resuming from chunk {start_chunk + 1}")
            # Load existing ABC data if available
            existing_abc_dir = output_dir / "abc"
            if existing_abc_dir.exists():
                existing_files = list(existing_abc_dir.glob("*.abc"))
                print(f"Found {len(existing_files)} existing ABC files")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            start_chunk = 0
    
    for chunk_idx in range(start_chunk, total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(args_list))
        chunk_args = args_list[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} "
              f"(files {start_idx + 1}-{end_idx} of {len(args_list)})")
        
        try:
            # Process chunk in parallel
            with Pool(processes=num_workers) as pool:
                # Use imap_unordered for better performance, but maintain order
                chunk_results = list(tqdm(
                    pool.imap(_convert_single_midi, chunk_args),
                    total=len(chunk_args),
                    desc=f"Chunk {chunk_idx + 1}/{total_chunks}"
                ))
            
            # Collect results from chunk
            for result in chunk_results:
                if result is not None:
                    abc_data.append(result)
                    conversion_stats['success'] += 1
                else:
                    conversion_stats['failed'] += 1
            
            # Force garbage collection between chunks
            import gc
            gc.collect()
            
            # Save checkpoint after each successful chunk
            checkpoint_data = {
                'last_chunk': chunk_idx,
                'success_count': conversion_stats['success'],
                'failed_count': conversion_stats['failed'],
                'processed_files': len(abc_data),
                'total_chunks': total_chunks
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Print progress
            progress = (chunk_idx + 1) / total_chunks * 100
            print(f"  Progress: {progress:.1f}% | "
                  f"Success: {conversion_stats['success']} | "
                  f"Failed: {conversion_stats['failed']}")
            print(f"  Checkpoint saved: chunk {chunk_idx + 1}/{total_chunks}")
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user at chunk {chunk_idx + 1}")
            print(f"Progress saved. You can resume by running the script again.")
            break
        
        except Exception as chunk_error:
            print(f"\nERROR processing chunk {chunk_idx + 1}: {type(chunk_error).__name__}: {chunk_error}")
            print(f"Error details:")
            import traceback
            traceback.print_exc()
            print(f"\nSaving checkpoint and continuing with next chunk...")
            
            # Save checkpoint even on error
            checkpoint_data = {
                'last_chunk': chunk_idx - 1,  # Mark previous chunk as last successful
                'success_count': conversion_stats['success'],
                'failed_count': conversion_stats['failed'],
                'processed_files': len(abc_data),
                'total_chunks': total_chunks,
                'error': str(chunk_error)
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Ask user if they want to continue (in automated mode, just continue)
            print(f"Continuing with next chunk...")
            continue
    
    print(f"\nConversion statistics:")
    print(f"  Successful: {conversion_stats['success']}")
    print(f"  Failed: {conversion_stats['failed']}")
    print(f"  Success rate: {conversion_stats['success'] / len(args_list) * 100:.1f}%")
    
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
    print(f"  Meets requirement: {'YES' if train_tokens >= MIN_TRAIN_TOKENS else 'NO'}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess LMD dataset')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of MIDI files to process (for testing)')
    parser.add_argument('--skip-conversion', action='store_true',
                        help='Skip MIDI to ABC conversion (use existing ABC files)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count - 1, current: {max(1, cpu_count() - 1)})')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of files to process per chunk (default: 1000, reduces memory usage)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint (automatically detected if checkpoint exists)')
    parser.add_argument('--clear-checkpoint', action='store_true',
                        help='Clear existing checkpoint and start from beginning')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Clear checkpoint if requested
    checkpoint_file = OUTPUT_DIR / "conversion_checkpoint.json"
    if args.clear_checkpoint and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint cleared. Starting from beginning.")
    
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
            num_workers=args.num_workers,
            chunk_size=args.chunk_size
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

