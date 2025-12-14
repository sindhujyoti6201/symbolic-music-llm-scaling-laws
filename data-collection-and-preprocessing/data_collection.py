#!/usr/bin/env python3
"""
Data Collection Script for Lakh MIDI Dataset
Downloads and extracts LMD-matched dataset for symbolic music LLM project.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
LMD_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
MATCH_SCORES_URL = "http://hog.ee.columbia.edu/craffel/lmd/match_scores.json"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class DownloadProgressBar:
    """Progress bar for file downloads."""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url: str, destination: Path, description: str = "File"):
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path where file should be saved
        description: Description for progress bar
    """
    if destination.exists():
        print(f"✓ {description} already exists: {destination}")
        return True
    
    print(f"Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, destination, DownloadProgressBar())
        print(f"✓ Successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {description}: {e}")
        return False


def extract_tar_gz(tar_path: Path, extract_to: Path):
    """
    Extract a tar.gz file to specified directory.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
    """
    if not tar_path.exists():
        print(f"✗ Tar file not found: {tar_path}")
        return False
    
    print(f"Extracting {tar_path.name} to {extract_to}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get total members for progress bar
            members = tar.getmembers()
            total_size = sum(m.size for m in members)
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_to)
                    pbar.update(member.size)
        
        print(f"✓ Successfully extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Error extracting {tar_path}: {e}")
        return False


def download_match_scores():
    """Download match_scores.json file."""
    # Store in data/ since it's input data, not a result
    match_scores_path = DATA_DIR / "match_scores.json"
    return download_file(MATCH_SCORES_URL, match_scores_path, "match_scores.json")


def download_lmd_matched():
    """
    Download and extract LMD-matched dataset.
    
    Returns:
        True if successful, False otherwise
    """
    # Download tar.gz file
    tar_path = DATA_DIR / "lmd_matched.tar.gz"
    if not download_file(LMD_URL, tar_path, "LMD-matched dataset"):
        return False
    
    # Extract to data directory
    extract_to = DATA_DIR / "lmd_matched"
    if extract_to.exists():
        print(f"✓ LMD-matched already extracted to {extract_to}")
        print("  (Delete this directory if you want to re-extract)")
        return True
    
    if not extract_tar_gz(tar_path, DATA_DIR):
        return False
    
    # Rename extracted folder if needed
    # The tar file might extract to a specific folder name
    extracted_folders = [d for d in DATA_DIR.iterdir() 
                        if d.is_dir() and 'lmd' in d.name.lower()]
    if extracted_folders and not extract_to.exists():
        # If extraction created a differently named folder, rename it
        if len(extracted_folders) == 1:
            extracted_folders[0].rename(extract_to)
            print(f"✓ Renamed extracted folder to {extract_to}")
    
    return True


def main():
    """Main function to download and extract LMD dataset."""
    print("=" * 60)
    print("Lakh MIDI Dataset (LMD) Download Script")
    print("=" * 60)
    print()
    
    # Check available disk space (optional - just a warning)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print()
    print("Note: LMD-matched is ~1.7GB compressed, ~2-3GB extracted")
    print()
    
    # Download match_scores.json
    print("Step 1: Downloading match_scores.json...")
    if not download_match_scores():
        print("Warning: Could not download match_scores.json")
        print("You can download it manually from:")
        print("  http://hog.ee.columbia.edu/craffel/lmd/match_scores.json")
        print("  Save it to:", DATA_DIR / "match_scores.json")
    print()
    
    # Download and extract LMD-matched
    print("Step 2: Downloading and extracting LMD-matched dataset...")
    if download_lmd_matched():
        print()
        print("=" * 60)
        print("✓ Dataset download and extraction complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Verify data in:", DATA_DIR / "lmd_matched")
        print("2. Check match_scores.json in:", DATA_DIR / "match_scores.json")
        print("3. Use music21 or midi2abc to convert MIDI files to ABC notation")
        return 0
    else:
        print()
        print("=" * 60)
        print("✗ Dataset download/extraction failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

