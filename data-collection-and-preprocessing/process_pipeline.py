#!/usr/bin/env python3
"""
Complete data processing pipeline: MIDI to ABC conversion, tokenization, and splitting.
"""

import sys
import json
import gc
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter
import random

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_to_abc import MIDIToABCConverter, convert_single_midi_worker
from tokenizer import MusicTokenizer


def convert_midi_to_abc_parallel(
    midi_files: List[Path],
    output_dir: Path,
    num_workers: int = 14,
    chunk_size: int = 500,
    max_files: Optional[int] = None
):
    """Convert MIDI files to ABC notation using parallel processing."""
    if max_files:
        midi_files = midi_files[:max_files]
    
    print(f"Converting {len(midi_files)} MIDI files to ABC notation...")
    print(f"Using {num_workers} parallel workers")
    print(f"Processing in chunks of {chunk_size} files\n")
    
    # Try different parallel processing methods
    USE_JOBLIB = False
    USE_MULTIPROCESSING = False
    
    try:
        from joblib import Parallel, delayed
        USE_JOBLIB = True
        print("joblib available - using parallel processing")
    except ImportError:
        print("WARNING: joblib not available - trying multiprocessing")
        try:
            from multiprocessing import Pool
            USE_MULTIPROCESSING = True
        except ImportError:
            print("WARNING: multiprocessing not available - using sequential processing")
    
    # Prepare arguments for processing
    args_list = [(str(midi_file), str(output_dir)) for midi_file in midi_files]
    
    # Process in chunks to prevent memory buildup
    abc_data = []
    conversion_stats = {'success': 0, 'failed': 0}
    
    total_chunks = (len(args_list) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(args_list))
        chunk_args = args_list[start_idx:end_idx]
        
        print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (files {start_idx + 1}-{end_idx})...")
        
        chunk_results = []
        
        # Process chunk - use joblib for parallel processing if available
        if USE_JOBLIB:
            try:
                chunk_results = Parallel(n_jobs=num_workers, backend='loky', verbose=0)(
                    delayed(convert_single_midi_worker)(args) for args in chunk_args
                )
                print(f"  Completed chunk {chunk_idx + 1}/{total_chunks}")
            except Exception as e:
                print(f"  joblib parallel failed ({e}), falling back to sequential")
                USE_JOBLIB = False
        
        if USE_MULTIPROCESSING and not USE_JOBLIB:
            try:
                with Pool(processes=num_workers) as pool:
                    chunk_results = pool.map(convert_single_midi_worker, chunk_args)
                print(f"  Completed chunk {chunk_idx + 1}/{total_chunks}")
            except Exception as e:
                print(f"  multiprocessing failed ({e}), falling back to sequential")
                USE_MULTIPROCESSING = False
        
        if not USE_JOBLIB and not USE_MULTIPROCESSING:
            # Sequential processing
            converter = MIDIToABCConverter()
            chunk_results = []
            
            for midi_file_str, output_dir_str in tqdm(chunk_args, desc=f"Chunk {chunk_idx + 1}/{total_chunks}", leave=False):
                midi_file = Path(midi_file_str)
                abc_str = converter.convert_midi_to_abc(midi_file)
                
                if abc_str:
                    abc_path = Path(output_dir_str) / "abc" / f"{midi_file.stem}.abc"
                    abc_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(abc_path, 'w') as f:
                            f.write(abc_str)
                        if abc_path.exists() and abc_path.stat().st_size > 0:
                            chunk_results.append((str(midi_file), abc_str))
                        else:
                            chunk_results.append(None)
                    except Exception as e:
                        chunk_results.append(None)
                else:
                    chunk_results.append(None)
                
                # Periodic garbage collection every 50 files
                if len(chunk_results) % 50 == 0:
                    gc.collect()
        
        # Collect results
        for result in chunk_results:
            if result is not None:
                abc_data.append(result)
                conversion_stats['success'] += 1
            else:
                conversion_stats['failed'] += 1
        
        # Garbage collection after each chunk
        gc.collect()
    
    print(f"\n{'='*60}")
    print("MIDI to ABC Conversion Complete")
    print(f"{'='*60}")
    print(f"Success: {conversion_stats['success']:,}")
    print(f"Failed: {conversion_stats['failed']:,}")
    print(f"Success rate: {conversion_stats['success']/(conversion_stats['success']+conversion_stats['failed'])*100:.1f}%")
    
    return abc_data, conversion_stats


def build_vocab_and_tokenize(
    abc_data: List[Tuple[str, str]],
    output_dir: Path,
    min_freq: int = 2
):
    """Build vocabulary and tokenize ABC strings."""
    print(f"\n{'='*60}")
    print("Building Vocabulary and Tokenizing")
    print(f"{'='*60}")
    
    # Extract ABC strings
    abc_strings = [abc_str for _, abc_str in abc_data]
    
    # Build tokenizer
    tokenizer = MusicTokenizer()
    tokenizer.build_vocab(abc_strings, min_freq=min_freq)
    
    # Tokenize all sequences
    print("\nTokenizing sequences...")
    tokenized_data = []
    for midi_path, abc_str in tqdm(abc_data, desc="Tokenizing"):
        token_ids = tokenizer.encode(abc_str)
        tokenized_data.append({
            'midi_path': midi_path,
            'abc': abc_str,
            'token_ids': token_ids,
            'length': len(token_ids)
        })
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.pkl"
    tokenizer.save(tokenizer_path)
    print(f"\nTokenizer saved to: {tokenizer_path}")
    
    return tokenizer, tokenized_data


def filter_sequences(
    tokenized_data: List[dict],
    min_length: int = 10,
    max_length: int = 5000
):
    """Filter sequences by length."""
    print(f"\n{'='*60}")
    print("Filtering Sequences")
    print(f"{'='*60}")
    
    original_count = len(tokenized_data)
    filtered_data = [
        item for item in tokenized_data
        if min_length <= item['length'] <= max_length
    ]
    
    print(f"Original sequences: {original_count:,}")
    print(f"Filtered sequences: {len(filtered_data):,}")
    print(f"Removed: {original_count - len(filtered_data):,} ({100*(original_count-len(filtered_data))/original_count:.1f}%)")
    
    return filtered_data


def create_splits(
    tokenized_data: List[dict],
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01
):
    """Create train/validation/test splits."""
    print(f"\n{'='*60}")
    print("Creating Data Splits")
    print(f"{'='*60}")
    
    # Shuffle data
    random.shuffle(tokenized_data)
    
    # Calculate split indices
    total = len(tokenized_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = tokenized_data[:train_end]
    val_data = tokenized_data[train_end:val_end]
    test_data = tokenized_data[val_end:]
    
    print(f"Train: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
    print(f"Validation: {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
    print(f"Test: {len(test_data):,} ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data


def save_splits(
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    output_dir: Path
):
    """Save data splits to JSON files."""
    print(f"\n{'='*60}")
    print("Saving Data Splits")
    print(f"{'='*60}")
    
    # Create directories
    (output_dir / "tokenized" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenized" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenized" / "test").mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "tokenized" / "train" / "data.json"
    val_path = output_dir / "tokenized" / "val" / "data.json"
    test_path = output_dir / "tokenized" / "test" / "data.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Train data saved to: {train_path}")
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Validation data saved to: {val_path}")
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved to: {test_path}")


def generate_statistics(tokenized_data: List[dict], output_dir: Path):
    """Generate and save data statistics."""
    print(f"\n{'='*60}")
    print("Generating Statistics")
    print(f"{'='*60}")
    
    lengths = [item['length'] for item in tokenized_data]
    
    stats = {
        'total_sequences': len(tokenized_data),
        'total_tokens': sum(lengths),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'median_length': sorted(lengths)[len(lengths)//2] if lengths else 0
    }
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    print(f"  Total sequences: {stats['total_sequences']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Average length: {stats['avg_length']:.1f}")
    print(f"  Length range: {stats['min_length']} - {stats['max_length']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MIDI files to tokenized data")
    parser.add_argument("--midi-dir", type=str, required=True,
                       help="Directory containing MIDI files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of MIDI files to process")
    parser.add_argument("--num-workers", type=int, default=14,
                       help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=500,
                       help="Chunk size for processing")
    parser.add_argument("--min-freq", type=int, default=2,
                       help="Minimum token frequency for vocabulary")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=5000,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Setup paths
    midi_dir = Path(args.midi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "abc").mkdir(parents=True, exist_ok=True)
    
    # Find MIDI files
    print(f"Searching for MIDI files in {midi_dir}...")
    midi_files = list(midi_dir.rglob("*.mid"))
    print(f"Found {len(midi_files):,} MIDI files")
    
    if args.max_files:
        midi_files = midi_files[:args.max_files]
        print(f"Limiting to {len(midi_files):,} files")
    
    # Step 1: Convert MIDI to ABC
    abc_data, conversion_stats = convert_midi_to_abc_parallel(
        midi_files,
        output_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        max_files=args.max_files
    )
    
    if len(abc_data) == 0:
        print("ERROR: No ABC files were successfully converted!")
        sys.exit(1)
    
    # Step 2: Build vocabulary and tokenize
    tokenizer, tokenized_data = build_vocab_and_tokenize(
        abc_data,
        output_dir,
        min_freq=args.min_freq
    )
    
    # Step 3: Filter sequences
    filtered_data = filter_sequences(
        tokenized_data,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Step 4: Create splits
    train_data, val_data, test_data = create_splits(filtered_data)
    
    # Step 5: Save splits
    save_splits(train_data, val_data, test_data, output_dir)
    
    # Step 6: Generate statistics
    generate_statistics(filtered_data, output_dir)
    
    print(f"\n{'='*60}")
    print("DATA PROCESSING PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

