#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Generate ground truth durations using Montreal Forced Aligner (MFA).

Uses MFA to align audio with text transcripts, then extracts phoneme/word
durations from TextGrid outputs.

Requirements:
    pip install montreal-forced-aligner
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

Usage:
    python generate_durations_mfa.py \
        --dataset-path ./data/tts-processed \
        --output-dir ./data/tts-processed \
        --acoustic-model english_us_arpa \
        --dictionary english_us_arpa
"""

import torch
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import subprocess
import tempfile
import shutil
from typing import List, Dict, Tuple


def prepare_mfa_corpus(dataset_index: List[Dict], dataset_path: Path, temp_dir: Path) -> Path:
    """
    Prepare corpus directory for MFA alignment.
    
    MFA expects:
    - corpus/
      - speaker1/
        - file1.wav
        - file1.lab (text transcript)
        - file2.wav
        - file2.lab
    
    Args:
        dataset_index: List of dataset entries with audio paths and text
        dataset_path: Root path to dataset
        temp_dir: Temporary directory for MFA corpus
    
    Returns:
        Path to prepared corpus directory
    """
    corpus_dir = temp_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    # Use single speaker directory (LJSpeech is single speaker)
    speaker_dir = corpus_dir / "speaker1"
    speaker_dir.mkdir(exist_ok=True)
    
    print(f"Preparing MFA corpus in {corpus_dir}")
    
    for item in tqdm(dataset_index, desc="Preparing corpus"):
        sample_id = item['id']
        text = item['text']
        
        # Find audio file (from original dataset)
        # Assuming wavs are in dataset_path/../../tts-datasets/ljspeech_subset_50/wavs/
        audio_source = dataset_path.parent / "tts-datasets" / "ljspeech_subset_50" / "wavs" / f"{sample_id}.wav"
        
        if not audio_source.exists():
            print(f"⚠ Audio not found: {audio_source}")
            continue
        
        # Copy audio
        audio_dest = speaker_dir / f"{sample_id}.wav"
        shutil.copy2(audio_source, audio_dest)
        
        # Write transcript (.lab file)
        lab_file = speaker_dir / f"{sample_id}.lab"
        with open(lab_file, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"✓ Corpus prepared with {len(list(speaker_dir.glob('*.wav')))} files")
    return corpus_dir


def run_mfa_alignment(
    corpus_dir: Path,
    output_dir: Path,
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
) -> Path:
    """
    Run MFA forced alignment.
    
    Args:
        corpus_dir: Path to prepared corpus
        output_dir: Output directory for TextGrids
        acoustic_model: Name of acoustic model to use
        dictionary: Name of pronunciation dictionary
    
    Returns:
        Path to directory containing TextGrid files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Running Montreal Forced Aligner")
    print("=" * 70)
    print(f"  Corpus: {corpus_dir}")
    print(f"  Acoustic model: {acoustic_model}")
    print(f"  Dictionary: {dictionary}")
    print(f"  Output: {output_dir}")
    
    # Run MFA alignment
    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--clean",  # Clean temporary files
        "--use_mp",  # Use multiprocessing
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nThis may take several minutes...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("✓ MFA alignment completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ MFA alignment failed:")
        print(f"  Return code: {e.returncode}")
        print(f"  STDOUT:\n{e.stdout}")
        print(f"  STDERR:\n{e.stderr}")
        raise RuntimeError("MFA alignment failed") from e
    
    return output_dir


def parse_textgrid(textgrid_path: Path) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """
    Parse TextGrid file to extract word and phone intervals using praatio.
    
    Args:
        textgrid_path: Path to TextGrid file
    
    Returns:
        Tuple of (word_intervals, phone_intervals)
        Each interval is (label, start_time, end_time)
    """
    from praatio import textgrid
    
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    
    word_intervals = []
    phone_intervals = []
    
    # MFA outputs "words" and "phones" tiers
    if "words" in tg.tierNames:
        words_tier = tg.getTier("words")
        for entry in words_tier.entries:
            label = entry.label.strip()
            if label and label not in ["sp", "sil", ""]:  # Skip silences
                word_intervals.append((label, entry.start, entry.end))
    
    if "phones" in tg.tierNames:
        phones_tier = tg.getTier("phones")
        for entry in phones_tier.entries:
            label = entry.label.strip()
            if label and label not in ["sp", "sil", "spn", ""]:  # Skip silences
                phone_intervals.append((label, entry.start, entry.end))
    
    return word_intervals, phone_intervals


def textgrid_to_durations(
    textgrid_path: Path,
    num_tokens: int,
    text: str,
    sample_rate: int = 24000,
    frame_shift: float = 0.0116,  # 12Hz framerate
) -> torch.Tensor:
    """
    Convert TextGrid alignments to token durations (in frames).
    
    Strategy:
    1. Use word-level alignments from TextGrid
    2. Map words to tokens based on text matching
    3. Accumulate phoneme durations within each word
    4. Distribute word duration across its tokens
    
    Args:
        textgrid_path: Path to TextGrid file
        num_tokens: Expected number of tokens
        text: Original text for matching
        sample_rate: Audio sample rate
        frame_shift: Time per frame (seconds)
    
    Returns:
        Tensor of durations [num_tokens]
    """
    word_intervals, phone_intervals = parse_textgrid(textgrid_path)
    
    if not word_intervals and not phone_intervals:
        print(f"⚠ No intervals found in {textgrid_path}")
        # Fallback: equal distribution
        total_duration = 1.0  # 1 second fallback
        frames_per_token = total_duration / frame_shift / num_tokens
        return torch.ones(num_tokens) * frames_per_token
    
    # Use word intervals if available, otherwise use phone intervals
    intervals = word_intervals if word_intervals else phone_intervals
    
    # Calculate total audio duration
    total_audio_duration = max(end for _, _, end in intervals) if intervals else 1.0
    
    # Initialize durations
    durations = torch.zeros(num_tokens)
    
    # If we have exact word count matching token count
    if len(intervals) == num_tokens:
        # Perfect match: one word per token
        for i, (label, start, end) in enumerate(intervals):
            duration_sec = end - start
            durations[i] = duration_sec / frame_shift
    else:
        # Need to map multiple intervals to tokens or vice versa
        # Strategy: distribute intervals proportionally
        
        # Calculate time per token
        time_per_token = total_audio_duration / num_tokens
        
        for i in range(num_tokens):
            token_start_time = i * time_per_token
            token_end_time = (i + 1) * time_per_token
            
            # Find all intervals overlapping with this token's time range
            token_duration = 0.0
            for label, start, end in intervals:
                # Calculate overlap
                overlap_start = max(token_start_time, start)
                overlap_end = min(token_end_time, end)
                
                if overlap_end > overlap_start:
                    token_duration += (overlap_end - overlap_start)
            
            # Convert to frames
            durations[i] = max(token_duration / frame_shift, 1.0)
    
    # Ensure all durations are positive integers
    durations = torch.clamp(durations, min=1.0)
    durations = torch.round(durations)
    
    # Adjust to match exact total if needed (from phoneme data)
    if phone_intervals:
        total_phone_duration = sum(end - start for _, start, end in phone_intervals)
        target_frames = total_phone_duration / frame_shift
        current_frames = durations.sum().item()
        
        if abs(current_frames - target_frames) > 1.0:
            # Scale to match
            scale_factor = target_frames / current_frames
            durations = torch.round(durations * scale_factor)
            durations = torch.clamp(durations, min=1.0)
    
    return durations


def generate_durations_mfa(
    dataset_path: str,
    output_dir: str,
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
):
    """
    Generate durations using Montreal Forced Aligner.
    
    Args:
        dataset_path: Path to dataset with final_index.json
        output_dir: Output directory
        acoustic_model: MFA acoustic model name
        dictionary: MFA dictionary name
    """
    dataset_path_obj = Path(dataset_path)
    output_path = Path(output_dir)
    
    # Load index
    index_file = dataset_path_obj / "final_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index not found: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        dataset_index = json.load(f)
    
    print("=" * 70)
    print("Generating Durations via Montreal Forced Aligner")
    print("=" * 70)
    print(f"  Dataset: {dataset_path_obj}")
    print(f"  Samples: {len(dataset_index)}")
    print(f"  Acoustic model: {acoustic_model}")
    print(f"  Dictionary: {dictionary}")
    
    # Create temporary directory for MFA processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Prepare corpus
        corpus_dir = prepare_mfa_corpus(dataset_index, dataset_path_obj, temp_path)
        
        # Step 2: Run MFA alignment
        textgrid_output = temp_path / "textgrids"
        run_mfa_alignment(corpus_dir, textgrid_output, acoustic_model, dictionary)
        
        # Step 3: Extract durations from TextGrids
        durations_dir = output_path / "durations_mfa"
        durations_dir.mkdir(parents=True, exist_ok=True)
        
        updated_index = []
        
        print("\nExtracting durations from TextGrids...")
        for item in tqdm(dataset_index, desc="Processing TextGrids"):
            sample_id = item['id']
            num_tokens = item['num_tokens']
            text = item['text']
            
            # Find TextGrid (MFA outputs to speaker1 subdirectory)
            textgrid_path = textgrid_output / "speaker1" / f"{sample_id}.TextGrid"
            
            if not textgrid_path.exists():
                print(f"⚠ TextGrid not found: {textgrid_path}")
                continue
            
            # Convert TextGrid to durations
            durations = textgrid_to_durations(textgrid_path, num_tokens, text)
            
            # Validate
            if durations.sum() < 1.0:
                print(f"⚠ Invalid durations for {sample_id}: sum={durations.sum()}")
                continue
            
            # Save
            duration_file = durations_dir / f"{sample_id}.pt"
            torch.save(durations, duration_file)
            
            # Update index
            item_copy = item.copy()
            item_copy['durations_path'] = f"durations_mfa/{sample_id}.pt"
            item_copy['avg_duration'] = durations.mean().item()
            item_copy['min_duration'] = durations.min().item()
            item_copy['max_duration'] = durations.max().item()
            item_copy['duration_method'] = 'mfa'
            
            updated_index.append(item_copy)
    
    # Save training index
    training_index_file = output_path / "training_index_mfa.json"
    with open(training_index_file, 'w', encoding='utf-8') as f:
        json.dump(updated_index, f, indent=2, ensure_ascii=False)
    
    # Stats
    avg_dur = sum(item['avg_duration'] for item in updated_index) / len(updated_index)
    
    print("\n" + "=" * 70)
    print("MFA Duration Generation Complete")
    print("=" * 70)
    print(f"  Processed: {len(updated_index)} samples")
    print(f"  Avg duration: {avg_dur:.2f} frames/token")
    print(f"  Method: Montreal Forced Aligner")
    print(f"\n  Output: {output_path}")
    print(f"  Training index: {training_index_file}")
    print("\n✓ High-quality MFA durations ready for training!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate durations using Montreal Forced Aligner"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset with final_index.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--acoustic-model',
        type=str,
        default='english_us_arpa',
        help='MFA acoustic model name (download with: mfa model download acoustic <name>)'
    )
    parser.add_argument(
        '--dictionary',
        type=str,
        default='english_us_arpa',
        help='MFA dictionary name (download with: mfa model download dictionary <name>)'
    )
    
    args = parser.parse_args()
    
    # Check MFA installation
    try:
        result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
        print(f"MFA version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("✗ Montreal Forced Aligner not found!")
        print("  Install with: pip install montreal-forced-aligner")
        print("  Or conda: conda install -c conda-forge montreal-forced-aligner")
        return
    
    generate_durations_mfa(
        args.dataset_path,
        args.output_dir,
        args.acoustic_model,
        args.dictionary,
    )


if __name__ == "__main__":
    main()
