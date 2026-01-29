#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Download and prepare small TTS dataset for testing.

Options:
1. LJSpeech (subset) - 100 samples (~5 minutes)
2. Common Voice (tiny) - 50 samples
"""

import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import json
import random


def download_ljspeech_subset(output_dir: str, num_samples: int = 100):
    """
    Download LJSpeech and take subset for quick testing.
    
    LJSpeech: 13,100 clips, single speaker (female), high quality
    Total: ~24 hours, but we'll use first 100 clips (~5 min)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading LJSpeech Dataset (Subset)")
    print("=" * 70)
    
    # LJSpeech URL
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    archive_path = output_path / "LJSpeech-1.1.tar.bz2"
    
    # Download if not exists
    if not archive_path.exists():
        print(f"\nDownloading from {url}")
        print("Size: ~2.6 GB (this will take a while...)")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, archive_path, reporthook=progress_hook)
        print("\n✓ Download complete!")
    else:
        print(f"\n✓ Archive already exists: {archive_path}")
    
    # Extract
    extract_path = output_path / "LJSpeech-1.1"
    if not extract_path.exists():
        print(f"\nExtracting to {extract_path}...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(output_path)
        print("✓ Extraction complete!")
    else:
        print(f"\n✓ Already extracted: {extract_path}")
    
    # Read metadata
    metadata_file = extract_path / "metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n✓ Total samples in LJSpeech: {len(lines)}")
    
    # Take subset
    subset_lines = lines[:num_samples]
    
    # Create subset directory
    subset_path = output_path / f"ljspeech_subset_{num_samples}"
    subset_path.mkdir(exist_ok=True)
    
    (subset_path / "wavs").mkdir(exist_ok=True)
    
    # Copy files and create index
    index = []
    
    print(f"\nCreating subset with {num_samples} samples...")
    for i, line in enumerate(subset_lines):
        parts = line.strip().split('|')
        file_id = parts[0]
        text = parts[2] if len(parts) > 2 else parts[1]
        
        # Copy wav file
        src_wav = extract_path / "wavs" / f"{file_id}.wav"
        dst_wav = subset_path / "wavs" / f"{file_id}.wav"
        
        if src_wav.exists():
            if not dst_wav.exists():
                import shutil
                shutil.copy2(src_wav, dst_wav)
            
            index.append({
                'id': file_id,
                'audio_path': f"wavs/{file_id}.wav",
                'text': text,
                'speaker': 'LJ',
            })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples}")
    
    # Save index
    index_file = subset_path / "index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Subset created: {subset_path}")
    print(f"  Samples: {len(index)}")
    print(f"  Index: {index_file}")
    
    # Stats
    total_chars = sum(len(item['text']) for item in index)
    avg_chars = total_chars / len(index)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(index)}")
    print(f"  Avg text length: {avg_chars:.1f} chars")
    print(f"  Speaker: Single (LJ - female)")
    print(f"  Sample rate: 22050 Hz")
    print(f"  Estimated duration: ~{len(index) * 3}s (~{len(index) * 3 / 60:.1f} min)")
    
    return subset_path


def download_common_voice_tiny(output_dir: str):
    """
    Alternative: Common Voice tiny subset (manual download required).
    """
    print("=" * 70)
    print("Common Voice Tiny Subset")
    print("=" * 70)
    print("\nCommon Voice requires manual download from:")
    print("https://commonvoice.mozilla.org/en/datasets")
    print("\nSteps:")
    print("1. Create account")
    print("2. Download English dataset")
    print("3. Extract and place in:", output_dir)
    print("\nFor quick testing, use LJSpeech instead (automated).")


def main():
    """Main download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TTS dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        default='ljspeech',
        choices=['ljspeech', 'commonvoice'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/tts-datasets',
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples for subset (LJSpeech only)'
    )
    
    args = parser.parse_args()
    
    if args.dataset == 'ljspeech':
        subset_path = download_ljspeech_subset(
            args.output_dir,
            args.num_samples
        )
        print("\n" + "=" * 70)
        print("✓ Dataset Ready!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Run preprocess_audio.py")
        print(f"     --dataset-path {subset_path}")
        print(f"2. Extract Llama features")
        print(f"3. Start training")
    
    elif args.dataset == 'commonvoice':
        download_common_voice_tiny(args.output_dir)


if __name__ == "__main__":
    main()
