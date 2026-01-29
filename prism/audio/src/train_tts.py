# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Training pipeline for End-to-End TTS with Duration Prediction.

Trains complete TTS projector with:
- Acoustic feature reconstruction loss
- Duration prediction loss
- Attention alignment loss
"""

# CRITICAL: Set CUDA device BEFORE importing torch
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Tesla P4

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

from projector import TTSProjector


@dataclass
class TTSTrainingConfig:
    """TTS training configuration."""
    
    # Model paths
    llm_path: str
    output_dir: str
    
    # Model architecture
    llm_hidden_size: int = 4096
    tts_hidden_size: int = 2048
    num_cross_attn_layers: int = 4
    num_attn_heads: int = 8
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Loss weights
    acoustic_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1
    cosine_loss_weight: float = 0.5
    
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
    
    def validate(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
        
        output_path = Path(self.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Training config validated")


class TTSDataset(Dataset):
    """
    Dataset for TTS training.
    
    Expected structure:
    - llm_hidden_states: pre-extracted LLM hidden states
    - acoustic_features: target acoustic features (mel-spec or audio tokens)
    - durations: ground truth durations (frames per token)
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
        index_file = self.data_path / "training_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_file}")
        
        with open(index_file) as f:
            self.index = json.load(f)
        
        if max_samples is not None:
            self.index = self.index[:max_samples]
        
        print(f"✓ Dataset loaded: {len(self.index)} samples")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.index[idx]
        
        # Load features - paths are relative to data_path
        llm_hidden_path = self.data_path / sample['llm_hidden_states_path']
        acoustic_path = self.data_path / sample['acoustic_features_path']
        durations_path = self.data_path / sample['durations_path']
        
        llm_hidden = torch.load(llm_hidden_path, map_location='cpu', weights_only=False)
        acoustic_features = torch.load(acoustic_path, map_location='cpu', weights_only=False)
        durations = torch.load(durations_path, map_location='cpu', weights_only=False)
        
        # Validate
        if torch.isnan(llm_hidden).any():
            raise RuntimeError(f"NaN in LLM hidden states: {llm_hidden_path}")
        
        if torch.isnan(acoustic_features).any():
            raise RuntimeError(f"NaN in acoustic features: {acoustic_path}")
        
        if torch.isnan(durations).any():
            raise RuntimeError(f"NaN in durations: {durations_path}")
        
        # Create attention mask (all ones since we don't have padding)
        attention_mask = torch.ones(llm_hidden.shape[0])
        
        return {
            'llm_hidden_states': llm_hidden,
            'acoustic_features': acoustic_features,
            'durations': durations,
            'attention_mask': attention_mask,
        }


class TTSTrainer:
    """TTS Projector Trainer."""
    
    def __init__(
        self,
        config: TTSTrainingConfig,
        train_dataset: TTSDataset,
        eval_dataset: Optional[TTSDataset] = None,
    ):
        self.config = config
        self.config.validate()
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Initialize model
        self.model = TTSProjector(
            llm_hidden_size=config.llm_hidden_size,
            tts_hidden_size=config.tts_hidden_size,
            num_cross_attn_layers=config.num_cross_attn_layers,
            num_attn_heads=config.num_attn_heads,
            dropout=config.dropout,
        )
        self.model.to(config.device)
        
        num_params = self.model.get_num_params()
        print(f"✓ Model initialized: {num_params:,} parameters")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        
        # Loss functions
        self.acoustic_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()
        
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
        print("Starting TTS Training")
        print("=" * 70)
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        self.model.train()
        
        accumulated_loss = 0.0
        
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device
                llm_hidden = batch['llm_hidden_states'].to(self.config.device)
                acoustic_target = batch['acoustic_features'].to(self.config.device)
                gt_durations = batch['durations'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)
                
                # Validate batch
                if torch.isnan(llm_hidden).any():
                    raise RuntimeError(f"NaN in batch LLM hidden at step {self.global_step}")
                
                # Forward pass
                outputs = self.model(
                    hidden_states=llm_hidden,
                    attention_mask=attention_mask,
                    target_length=acoustic_target.shape[1],
                    ground_truth_durations=gt_durations,
                )
                
                acoustic_pred = outputs['acoustic_features']
                duration_pred = outputs['predicted_durations']
                
                # Validate shapes
                if acoustic_pred.shape != acoustic_target.shape:
                    raise RuntimeError(
                        f"Shape mismatch: pred={acoustic_pred.shape}, "
                        f"target={acoustic_target.shape}"
                    )
                
                # Compute losses
                
                # 1. Acoustic reconstruction loss (MSE + Cosine)
                acoustic_mse = self.acoustic_loss(acoustic_pred, acoustic_target)
                acoustic_cosine_sim = torch.nn.functional.cosine_similarity(
                    acoustic_pred, acoustic_target, dim=-1
                ).mean()
                acoustic_cosine_loss = 1.0 - acoustic_cosine_sim
                
                acoustic_loss = acoustic_mse + self.config.cosine_loss_weight * acoustic_cosine_loss
                
                # 2. Duration prediction loss
                duration_loss_val = self.duration_loss(duration_pred, gt_durations)
                
                # 3. Combined loss
                total_loss = (
                    self.config.acoustic_loss_weight * acoustic_loss +
                    self.config.duration_loss_weight * duration_loss_val
                )
                
                # Check for NaN
                if torch.isnan(total_loss):
                    raise RuntimeError(
                        f"NaN loss at step {self.global_step}: "
                        f"acoustic={acoustic_loss.item():.4f}, "
                        f"duration={duration_loss_val.item():.4f}"
                    )
                
                # Scale for gradient accumulation
                total_loss = total_loss / self.config.gradient_accumulation_steps
                
                # Backward
                total_loss.backward()
                
                accumulated_loss += total_loss.item()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Check gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            raise RuntimeError(f"NaN gradient in {name} at step {self.global_step}")
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        lr = self.scheduler.get_last_lr()[0]
                        
                        # Log metrics
                        self.writer.add_scalar('train/loss_total', avg_loss, self.global_step)
                        self.writer.add_scalar('train/loss_acoustic_mse', acoustic_mse.item(), self.global_step)
                        self.writer.add_scalar('train/loss_duration', duration_loss_val.item(), self.global_step)
                        self.writer.add_scalar('train/cosine_similarity', acoustic_cosine_sim.item(), self.global_step)
                        self.writer.add_scalar('train/lr', lr, self.global_step)
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'ac_mse': f'{acoustic_mse.item():.4f}',
                            'dur': f'{duration_loss_val.item():.4f}',
                            'cos': f'{acoustic_cosine_sim.item():.3f}',
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
        """Evaluate on validation set."""
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
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                llm_hidden = batch['llm_hidden_states'].to(self.config.device)
                acoustic_target = batch['acoustic_features'].to(self.config.device)
                gt_durations = batch['durations'].to(self.config.device)
                
                outputs = self.model(
                    hidden_states=llm_hidden,
                    target_length=acoustic_target.shape[1],
                    ground_truth_durations=gt_durations,
                )
                
                acoustic_pred = outputs['acoustic_features']
                
                loss = self.acoustic_loss(acoustic_pred, acoustic_target)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"\n  Eval loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir)
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
        }
        
        if is_final:
            checkpoint_path = output_dir / "checkpoint_final.pt"
        else:
            checkpoint_path = output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
        
        # Save projector only
        projector_path = output_dir / f"tts_projector_step_{self.global_step}.pt"
        self.model.save_pretrained(str(projector_path))


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = TTSTrainingConfig(
        llm_path="unused",  # Not needed, we have precomputed features
        output_dir=args.output_dir,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_steps=args.max_steps,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        device=args.device,
    )
    
    print("=" * 70)
    print("End-to-End TTS Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accum: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Device: {config.device}")
    
    # Load dataset
    train_dataset = TTSDataset(args.dataset_path)
    
    # Initialize trainer
    trainer = TTSTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
