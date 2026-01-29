#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Training pipeline for Audio Projection Adapter.

Implements full training loop with:
- Dataset loading and validation
- Gradient accumulation
- Learning rate scheduling
- Checkpoint management
- TensorBoard logging
- Distributed training support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import json
from tqdm import tqdm

from projector import AudioProjector


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model paths
    llm_path: str
    tts_path: str
    output_dir: str
    
    # Model architecture
    llm_hidden_size: int = 4096
    tts_hidden_size: int = 2048
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Device
    device: str = "cpu"
    mixed_precision: bool = False
    
    # Validation
    eval_samples: int = 100
    
    def validate(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
        
        if self.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {self.max_steps}")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"Invalid gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        
        if not Path(self.llm_path).exists():
            raise FileNotFoundError(f"LLM path not found: {self.llm_path}")
        
        if not Path(self.tts_path).exists():
            raise FileNotFoundError(f"TTS path not found: {self.tts_path}")
        
        output_path = Path(self.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Training config validated")


class AudioTextDataset(Dataset):
    """
    Dataset for audio-text pairs.
    
    Expected structure:
    - text: input text
    - llm_hidden_states: pre-extracted LLM hidden states
    - tts_features: target TTS features
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Load dataset index
        index_file = self.data_path / "index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_file}")
        
        with open(index_file) as f:
            self.index = json.load(f)
        
        if max_samples is not None:
            self.index = self.index[:max_samples]
        
        self.validate_dataset()
        
        print(f"✓ Dataset loaded: {len(self.index)} samples")
    
    def validate_dataset(self):
        """Validate dataset structure."""
        if not isinstance(self.index, list):
            raise ValueError("Dataset index must be a list")
        
        if len(self.index) == 0:
            raise ValueError("Dataset is empty")
        
        # Check first sample structure
        sample = self.index[0]
        required_keys = ['text', 'llm_hidden_states_path', 'tts_features_path']
        
        for key in required_keys:
            if key not in sample:
                raise ValueError(f"Dataset sample missing key: {key}")
        
        print(f"  Samples: {len(self.index)}")
        print(f"  Keys: {list(sample.keys())}")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.index[idx]
        
        # Load pre-extracted features
        llm_hidden_path = self.data_path / sample['llm_hidden_states_path']
        tts_features_path = self.data_path / sample['tts_features_path']
        
        if not llm_hidden_path.exists():
            raise FileNotFoundError(f"LLM hidden states not found: {llm_hidden_path}")
        
        if not tts_features_path.exists():
            raise FileNotFoundError(f"TTS features not found: {tts_features_path}")
        
        llm_hidden = torch.load(llm_hidden_path, map_location='cpu')
        tts_features = torch.load(tts_features_path, map_location='cpu')
        
        # Validate tensors
        if torch.isnan(llm_hidden).any():
            raise RuntimeError(f"NaN in LLM hidden states: {llm_hidden_path}")
        
        if torch.isnan(tts_features).any():
            raise RuntimeError(f"NaN in TTS features: {tts_features_path}")
        
        return {
            'llm_hidden_states': llm_hidden,
            'tts_features': tts_features,
            'text': sample['text'],
        }


class Trainer:
    """Audio Projection Adapter Trainer."""
    
    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: AudioTextDataset,
        eval_dataset: Optional[AudioTextDataset] = None,
    ):
        self.config = config
        self.config.validate()
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Initialize model
        self.model = AudioProjector(
            llm_hidden_size=config.llm_hidden_size,
            tts_hidden_size=config.tts_hidden_size,
            dropout=config.dropout,
        )
        self.model.to(config.device)
        
        # Validate model
        num_params = self.model.get_num_params()
        if num_params == 0:
            raise RuntimeError("Model has zero parameters")
        
        print(f"✓ Model initialized: {num_params:,} parameters")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        
        # Loss function: Hybrid MSE + Cosine Similarity
        self.mse_loss = nn.MSELoss()
        self.mse_weight = 1.0
        self.cosine_weight = 0.5  # Weight for cosine similarity component
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1,
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(Path(config.output_dir) / "logs"))
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        print("✓ Trainer initialized")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # CPU only
            pin_memory=False,
        )
        
        self.model.train()
        
        accumulated_loss = 0.0
        optimizer_step = 0
        
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device
                llm_hidden = batch['llm_hidden_states'].to(self.config.device)
                tts_target = batch['tts_features'].to(self.config.device)
                
                # Validate batch
                if torch.isnan(llm_hidden).any():
                    raise RuntimeError(f"NaN in batch LLM hidden states at step {self.global_step}")
                
                if torch.isnan(tts_target).any():
                    raise RuntimeError(f"NaN in batch TTS features at step {self.global_step}")
                
                # Forward pass
                projected = self.model(llm_hidden)
                
                # Validate sequence length alignment
                if projected.shape[1] != tts_target.shape[1]:
                    raise RuntimeError(
                        f"Sequence length mismatch at step {self.global_step}: "
                        f"projected={projected.shape[1]}, target={tts_target.shape[1]}. "
                        f"Consider using pooling or length regulator."
                    )
                
                # Compute hybrid loss: MSE + Cosine Similarity
                mse_loss = self.mse_loss(projected, tts_target)
                
                # Cosine similarity loss (1 - cosine_sim, so higher similarity = lower loss)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    projected, tts_target, dim=-1
                ).mean()
                cosine_loss = 1.0 - cosine_sim
                
                # Combined loss
                loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
                
                # Check for NaN loss
                if torch.isnan(loss):
                    raise RuntimeError(
                        f"NaN loss at step {self.global_step}: "
                        f"mse={mse_loss.item():.4f}, cosine_sim={cosine_sim.item():.4f}"
                    )
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                accumulated_loss += loss.item()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Check for NaN gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            raise RuntimeError(f"NaN gradient in {name} at step {self.global_step}")
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    optimizer_step += 1
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        lr = self.scheduler.get_last_lr()[0]
                        
                        # Log detailed metrics
                        self.writer.add_scalar('train/loss_total', avg_loss, self.global_step)
                        self.writer.add_scalar('train/loss_mse', mse_loss.item(), self.global_step)
                        self.writer.add_scalar('train/loss_cosine', cosine_loss.item(), self.global_step)
                        self.writer.add_scalar('train/cosine_similarity', cosine_sim.item(), self.global_step)
                        self.writer.add_scalar('train/lr', lr, self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'mse': f'{mse_loss.item():.4f}',
                            'cos_sim': f'{cosine_sim.item():.3f}',
                            'lr': f'{lr:.2e}',
                        })
                        
                        accumulated_loss = 0.0
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0 and self.eval_dataset is not None:
                        eval_loss = self.evaluate()
                        self.writer.add_scalar('eval/loss', eval_loss, self.global_step)
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint(is_best=True)
                        
                        self.model.train()
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                    
                    # Check if training is complete
                    if self.global_step >= self.config.max_steps:
                        break
            
            if self.global_step >= self.config.max_steps:
                break
        
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint(is_final=True)
        
        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"  Total steps: {self.global_step}")
        print(f"  Best eval loss: {self.best_eval_loss:.4f}")
    
    def evaluate(self) -> float:
        """Evaluate model on validation set with hybrid loss."""
        if self.eval_dataset is None:
            return 0.0
        
        self.model.eval()
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        total_loss = 0.0
        total_mse = 0.0
        total_cosine_sim = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                llm_hidden = batch['llm_hidden_states'].to(self.config.device)
                tts_target = batch['tts_features'].to(self.config.device)
                
                projected = self.model(llm_hidden)
                
                # Compute hybrid loss
                mse_loss = self.mse_loss(projected, tts_target)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    projected, tts_target, dim=-1
                ).mean()
                cosine_loss = 1.0 - cosine_sim
                
                loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_cosine_sim += cosine_sim.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_cosine_sim = total_cosine_sim / num_batches
        
        print(f"\n  Eval metrics:")
        print(f"    Total loss: {avg_loss:.4f}")
        print(f"    MSE loss: {avg_mse:.4f}")
        print(f"    Cosine similarity: {avg_cosine_sim:.3f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': {
                'llm_hidden_size': self.config.llm_hidden_size,
                'tts_hidden_size': self.config.tts_hidden_size,
                'dropout': self.config.dropout,
            }
        }
        
        # Save regular checkpoint
        if is_final:
            checkpoint_path = output_dir / "checkpoint_final.pt"
        else:
            checkpoint_path = output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        # Save projector only
        projector_path = output_dir / f"projector_step_{self.global_step}.pt"
        self.model.save_pretrained(str(projector_path))


def main():
    """Main training entry point."""
    
    # Configuration
    config = TrainingConfig(
        llm_path="./data/audio-projection-models/llama-3.1-8b-instruct",
        tts_path="./data/audio-projection-models/qwen3-tts-1.7b",
        output_dir="./data/audio-projection-models/checkpoints",
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_steps=100000,
        device="cpu",
    )
    
    print("=" * 70)
    print("Audio Projection Adapter Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LLM: {config.llm_path}")
    print(f"  TTS: {config.tts_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Device: {config.device}")
    
    # Note: Dataset needs to be prepared separately
    print("\n⚠ Dataset preparation required!")
    print("  Run data preparation script first to extract features")
    print("  Expected dataset structure:")
    print("    - index.json")
    print("    - llm_hidden_states/")
    print("    - tts_features/")


if __name__ == "__main__":
    main()
