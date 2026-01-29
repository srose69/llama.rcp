#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Extract mel-spectrograms from audio files.

Converts WAV -> Mel-spectrogram for TTS training.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import json
from tqdm import tqdm
import argparse


class MelSpectrogramExtractor:
    """Extract mel-spectrograms compatible with TTS models."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        target_dim: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.target_dim = target_dim
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Projection to target dimension
        self.projection = torch.nn.Linear(n_mels, target_dim)
        torch.nn.init.xavier_uniform_(self.projection.weight)
    
    def extract(self, audio_path: str) -> torch.Tensor:
        """
        Extract mel-spectrogram from audio file.
        
        Returns:
            mel: [num_frames, target_dim] tensor
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)  # [1, n_mels, time]
        mel = self.amplitude_to_db(mel)
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        # Transpose and project
        mel = mel.squeeze(0).transpose(0, 1)  # [time, n_mels]
        mel = self.projection(mel)  # [time, target_dim]
        
        return mel
    
    def get_duration_seconds(self, num_frames: int) -> float:
        """Convert frame count to seconds."""
        return num_frames * self.hop_length / self.sample_rate


def preprocess_dataset(
    dataset_path: str,
    output_dir: str,
    target_dim: int = 2048,
):
    """
    Preprocess audio dataset to mel-spectrograms.
    
    Args:
        dataset_path: Path to dataset with index.json
        output_dir: Output directory for processed features
        target_dim: Target feature dimension
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output subdirectories
    acoustic_dir = output_path / "acoustic_features"
    acoustic_dir.mkdir(exist_ok=True)
    
    # Load dataset index
    index_file = dataset_path / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index not found: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        dataset_index = json.load(f)
    
    print("=" * 70)
    print("Preprocessing Audio Dataset")
    print("=" * 70)
    print(f"  Dataset: {dataset_path}")
    print(f"  Samples: {len(dataset_index)}")
    print(f"  Output: {output_path}")
    print(f"  Target dim: {target_dim}")
    
    # Initialize extractor
    extractor = MelSpectrogramExtractor(target_dim=target_dim)
    
    # Process each sample
    processed_index = []
    
    for item in tqdm(dataset_index, desc="Processing audio"):
        sample_id = item['id']
        audio_path = dataset_path / item['audio_path']
        
        if not audio_path.exists():
            print(f"\n⚠ Audio not found: {audio_path}")
            continue
        
        try:
            # Extract mel-spectrogram
            mel_features = extractor.extract(str(audio_path))
            
            # Validate
            if torch.isnan(mel_features).any():
                print(f"\n⚠ NaN in features for {sample_id}")
                continue
            
            # Save
            feature_file = acoustic_dir / f"{sample_id}.pt"
            torch.save(mel_features, feature_file)
            
            # Update index
            processed_index.append({
                'id': sample_id,
                'text': item['text'],
                'acoustic_features_path': f"acoustic_features/{sample_id}.pt",
                'num_frames': mel_features.shape[0],
                'duration_sec': extractor.get_duration_seconds(mel_features.shape[0]),
                'feature_dim': mel_features.shape[1],
            })
            
        except Exception as e:
            print(f"\n✗ Error processing {sample_id}: {str(e)}")
            continue
    
    # Save processed index
    processed_index_file = output_path / "processed_index.json"
    with open(processed_index_file, 'w', encoding='utf-8') as f:
        json.dump(processed_index, f, indent=2, ensure_ascii=False)
    
    # Stats
    total_frames = sum(item['num_frames'] for item in processed_index)
    total_duration = sum(item['duration_sec'] for item in processed_index)
    avg_frames = total_frames / len(processed_index)
    
    print("\n" + "=" * 70)
    print("Preprocessing Complete")
    print("=" * 70)
    print(f"  Processed: {len(processed_index)} samples")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total duration: {total_duration:.1f}s (~{total_duration/60:.1f} min)")
    print(f"  Avg frames/sample: {avg_frames:.1f}")
    print(f"  Feature dim: {target_dim}")
    print(f"\n  Output: {output_path}")
    print(f"  Index: {processed_index_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio to mel-spectrograms")
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset with index.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed features'
    )
    parser.add_argument(
        '--target-dim',
        type=int,
        default=2048,
        help='Target feature dimension'
    )
    
    args = parser.parse_args()
    
    preprocess_dataset(
        args.dataset_path,
        args.output_dir,
        args.target_dim,
    )


if __name__ == "__main__":
    main()
