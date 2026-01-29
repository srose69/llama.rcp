#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Phase 2: Train MicroMoE Routers via Gradient Descent

Workers are FROZEN (already trained).
MicroMoE routers learn which workers to activate for each input.

Integration:
- "Heads" of MicroMoE routers replace lower model's embedding
- "Feet" of pipes replace upper model's lm_head
- Train routers to minimize reconstruction error

Usage:
    python train_mmoe.py \
        --trained-pipes trained_pipes \
        --upper-model /path/to/llama \
        --lower-model /path/to/liquidai \
        --epochs 100 \
        --batch-size 32 \
        --output-dir final_pipes
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import sys


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from build_pipes import ThreePipeArchitecture


def load_trained_pipes(trained_pipes_dir, device='cpu'):
    """
    Load pipes with trained workers (frozen).
    """
    pipes_path = Path(trained_pipes_dir)
    
    # Load architecture config
    with open(pipes_path.parent / 'builded_pipes' / 'architecture.json', 'r') as f:
        architecture = json.load(f)
    
    # Create model
    model = ThreePipeArchitecture(architecture).to(device)
    
    # Load trained worker weights for each pipe
    for pipe_name in ['pipe1', 'pipe2', 'pipe3']:
        pipe_file = pipes_path / f'{pipe_name}_trained.pt'
        if pipe_file.exists():
            pipe_state = torch.load(pipe_file, map_location=device)
            pipe_obj = getattr(model, pipe_name)
            pipe_obj.load_state_dict(pipe_state, strict=False)
    
    # Freeze all workers
    model.freeze_all_workers()
    
    return model


def extract_hidden_states_batch(model, texts, tokenizer, device='cpu'):
    """
    Extract hidden states from model for a batch of texts.
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last hidden state, mean over sequence
        hidden_states = outputs.hidden_states[-1].mean(dim=1)
    
    return hidden_states


def generate_training_data(upper_model, lower_model, num_samples=10000, batch_size=32, device='cpu'):
    """
    Generate training data for MicroMoE router training.
    
    Returns:
        X_upper: [num_samples, dim_upper] - input to pipes
        Z_lower: [num_samples, dim_lower] - target output from pipes
    """
    from transformers import AutoTokenizer
    
    print(f"\nGenerating {num_samples} training samples for router training...")
    
    # Get tokenizers
    upper_tokenizer = AutoTokenizer.from_pretrained(upper_model.config._name_or_path)
    lower_tokenizer = AutoTokenizer.from_pretrained(lower_model.config._name_or_path)
    
    if upper_tokenizer.pad_token is None:
        upper_tokenizer.pad_token = upper_tokenizer.eos_token
    if lower_tokenizer.pad_token is None:
        lower_tokenizer.pad_token = lower_tokenizer.eos_token
    
    # Generate diverse texts
    texts = [
        f"Training sample {i}: The model learns to map representations between different architectures."
        for i in range(num_samples)
    ]
    
    X_upper_list = []
    Z_lower_list = []
    
    upper_model.eval()
    lower_model.eval()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting states"):
        batch_texts = texts[i:i+batch_size]
        
        # Upper model hidden states
        X_batch = extract_hidden_states_batch(upper_model, batch_texts, upper_tokenizer, device)
        X_upper_list.append(X_batch.cpu())
        
        # Lower model target states
        Z_batch = extract_hidden_states_batch(lower_model, batch_texts, lower_tokenizer, device)
        Z_lower_list.append(Z_batch.cpu())
    
    X_upper = torch.cat(X_upper_list, dim=0)
    Z_lower = torch.cat(Z_lower_list, dim=0)
    
    print(f"✓ Generated data: X_upper={X_upper.shape}, Z_lower={Z_lower.shape}")
    
    return X_upper, Z_lower


def train_routers(model, train_loader, epochs=100, lr=0.001, device='cpu'):
    """
    Train MicroMoE routers via gradient descent.
    
    Workers are FROZEN.
    Only router parameters are updated.
    """
    # Get trainable parameters (only routers)
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    model.train()
    
    print(f"\nTraining MicroMoE routers for {epochs} epochs...")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, Z_batch in pbar:
            X_batch = X_batch.to(device)
            Z_batch = Z_batch.to(device)
            
            # Forward pass through pipes with MicroMoE routing
            output = model(X_batch, use_routers=True)
            
            # Loss: MSE between pipe output and target
            loss = F.mse_loss(output, Z_batch)
            
            # Backward pass (only routers update)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f} - Best: {best_loss:.6f}")
    
    print(f"\n✓ Training complete. Best loss: {best_loss:.6f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Train MicroMoE Routers via Gradient Descent'
    )
    parser.add_argument(
        '--trained-pipes',
        type=str,
        required=True,
        help='Directory with trained pipes from Phase 1'
    )
    parser.add_argument(
        '--upper-model',
        type=str,
        required=True,
        help='Path to upper model (Llama)'
    )
    parser.add_argument(
        '--lower-model',
        type=str,
        required=True,
        help='Path to lower model (LiquidAI)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of training samples'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='final_pipes',
        help='Output directory for final trained model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for training'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 2: TRAIN MICROMOE ROUTERS (GRADIENT DESCENT)")
    print("=" * 70)
    print()
    
    device = torch.device(args.device)
    
    # Load trained pipes (workers frozen)
    print("🔄 Loading trained pipes...")
    model = load_trained_pipes(args.trained_pipes, device)
    print(f"✓ Loaded 3-Pipe architecture")
    print(f"  Workers: FROZEN")
    print(f"  Routers: TRAINABLE")
    print()
    
    # Load models for data generation
    print("🔄 Loading models for data extraction...")
    upper_model = AutoModelForCausalLM.from_pretrained(
        args.upper_model,
        torch_dtype=torch.float32,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    lower_model = AutoModelForCausalLM.from_pretrained(
        args.lower_model,
        torch_dtype=torch.float32,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    print()
    
    # Generate training data
    X_upper, Z_lower = generate_training_data(
        upper_model, lower_model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Create DataLoader
    dataset = TensorDataset(X_upper, Z_lower)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train routers
    model = train_routers(
        model, train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Save final model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving final trained model...")
    model.save_architecture(output_path)
    
    # Save training metadata
    training_info = {
        'phase': 2,
        'workers': 'frozen (trained in Phase 1)',
        'routers': 'trained (gradient descent)',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_samples': args.num_samples,
        'upper_model': args.upper_model,
        'lower_model': args.lower_model,
        'status': 'READY FOR DEPLOYMENT'
    }
    
    with open(output_path / 'training_complete.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print()
    print("=" * 70)
    print("✅ MICROMOE ROUTER TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("📦 Final model saved to:", output_path)
    print()
    print("🎯 Model is ready for deployment:")
    print("  - Workers: FROZEN (trained via least squares)")
    print("  - Routers: TRAINED (sparse 30% activation)")
    print("  - Integration: Insert between upper.lm_head and lower.embedding")
    print()
    print("Usage in inference:")
    print("  1. Remove upper model's lm_head")
    print("  2. Remove lower model's embedding")
    print("  3. Insert 3-Pipe architecture between them")
    print("  4. Forward: upper → pipes (with MicroMoE) → lower")
    print()


if __name__ == '__main__':
    main()
