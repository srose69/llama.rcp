#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Build 3-Pipe parallel architecture with MicroMoE routers.

Creates trainable pipe structures with:
- 3 parallel pipes (Pipe 1, Pipe 2, Pipe 3)
- 512 workers per pipe (1536 total)
- MicroMoE router before each pipe
- Rigid mapping from upper to lower model

Each worker is frozen after one-shot training.
Each router learns which workers to activate (sparse 30% activation).

Usage:
    python build_pipes.py \
        --config calculated_pipes/architecture.json \
        --output-dir builded_pipes
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroMoERouter(nn.Module):
    """
    Micro Mixture of Experts Router.
    
    Learns which workers to activate for each input.
    Each pipe has its own independent router.
    
    Args:
        input_dim: Input dimension (e.g., 4096 from upper model)
        num_workers: Total workers in pipe (always 512)
        top_k: Number of workers to activate (default: 154 = 30%)
    """
    def __init__(self, input_dim, num_workers=512, top_k=154):
        super().__init__()
        self.input_dim = input_dim
        self.num_workers = num_workers
        self.top_k = top_k
        
        # Gating network: learns worker selection
        self.gate = nn.Linear(input_dim, num_workers)
        
        # Initialize with small random weights
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, x):
        """
        Compute gating scores and select top-k workers.
        
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            weights: [batch, top_k] - normalized weights for selected workers
            indices: [batch, top_k] - worker IDs to activate
        """
        # Compute gating scores
        gate_logits = self.gate(x)  # [batch, num_workers]
        
        # Select top-k workers
        scores, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        # scores: [batch, top_k], indices: [batch, top_k]
        
        # Normalize scores to weights
        weights = F.softmax(scores, dim=-1)  # [batch, top_k]
        
        return weights, indices
    
    def get_gate_weights(self):
        """Return gate network weights for inspection."""
        return self.gate.weight.data, self.gate.bias.data


class PipeWorker(nn.Module):
    """
    Single worker in a pipe.
    
    Performs: output_chunk = input_chunk @ tile.T
    
    Worker is frozen after one-shot least squares training.
    """
    def __init__(self, tile_shape, worker_id, pipe_id):
        super().__init__()
        self.tile_shape = tile_shape
        self.worker_id = worker_id
        self.pipe_id = pipe_id
        
        # Initialize tile with small random values
        # Will be trained via least squares and then frozen
        tile = torch.randn(tile_shape) * 0.01
        self.tile = nn.Parameter(tile, requires_grad=False)  # Frozen by default
    
    def forward(self, x_chunk):
        """
        Apply tile transformation to input chunk.
        
        Args:
            x_chunk: [batch, input_dim] or [batch, K]
        
        Returns:
            output_chunk: [batch, output_dim] or [batch, N]
        """
        return x_chunk @ self.tile.T
    
    def set_tile(self, tile_data):
        """Set tile data from trained weights."""
        self.tile.data = tile_data
        self.tile.requires_grad = False  # Ensure frozen
    
    def freeze(self):
        """Freeze this worker (called after training)."""
        self.tile.requires_grad = False


class Pipe(nn.Module):
    """
    Single pipe with 512 workers and MicroMoE router.
    
    All workers are independent experts.
    Router learns which workers to activate.
    """
    def __init__(self, pipe_id, pipe_config, input_dim, output_dim):
        super().__init__()
        self.pipe_id = pipe_id
        self.pipe_config = pipe_config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_workers = 512
        self.top_k = 154  # 30% sparse activation
        
        # MicroMoE Router
        self.router = MicroMoERouter(
            input_dim=input_dim,
            num_workers=self.num_workers,
            top_k=self.top_k
        )
        
        # Workers
        tile_shape = pipe_config['tile_shape']
        self.workers = nn.ModuleList([
            PipeWorker(tile_shape, worker_id, pipe_id)
            for worker_id in range(self.num_workers)
        ])
    
    def forward(self, x, use_router=True):
        """
        Forward pass through pipe.
        
        Args:
            x: Input tensor [batch, input_dim]
            use_router: If True, use sparse activation. If False, use all workers.
        
        Returns:
            output: [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        if use_router:
            # Sparse activation via MicroMoE
            weights, indices = self.router(x)  # [batch, top_k], [batch, top_k]
            
            output = torch.zeros(batch_size, self.output_dim, device=x.device)
            
            # Activate only selected workers
            for b in range(batch_size):
                for k in range(self.top_k):
                    worker_id = indices[b, k].item()
                    weight = weights[b, k]
                    
                    # Extract input chunk for this worker
                    # TODO: Implement proper chunk extraction based on worker metadata
                    worker_input = x[b:b+1]  # Placeholder
                    
                    # Apply worker
                    worker_output = self.workers[worker_id](worker_input)
                    
                    # Accumulate weighted output
                    output[b] += weight * worker_output[0]
        else:
            # Dense activation: all workers active
            output = torch.zeros(batch_size, self.output_dim, device=x.device)
            
            for worker_id in range(self.num_workers):
                worker_input = x  # Placeholder
                worker_output = self.workers[worker_id](worker_input)
                output += worker_output / self.num_workers
        
        return output
    
    def freeze_workers(self):
        """Freeze all workers in this pipe."""
        for worker in self.workers:
            worker.freeze()
    
    def get_worker_metadata(self):
        """Return metadata for all workers."""
        return [
            {
                'worker_id': w.worker_id,
                'pipe_id': w.pipe_id,
                'tile_shape': w.tile_shape
            }
            for w in self.workers
        ]


class ThreePipeArchitecture(nn.Module):
    """
    Complete 3-Pipe parallel architecture.
    
    - Pipe 1: [512×512×512] - cubic, pattern A
    - Pipe 2: [512×512×512] - cubic, pattern B
    - Pipe 3: [512×N×K] - flexible, remainder coverage
    
    All pipes process input simultaneously (parallel).
    Outputs are combined (sum or weighted).
    """
    def __init__(self, architecture_config):
        super().__init__()
        self.architecture = architecture_config
        
        input_dim = architecture_config['upper_model']['output_dim']
        output_dim = architecture_config['lower_model']['input_dim']
        
        # Create 3 parallel pipes
        self.pipe1 = Pipe(1, architecture_config['pipes']['pipe1'], input_dim, output_dim)
        self.pipe2 = Pipe(2, architecture_config['pipes']['pipe2'], input_dim, output_dim)
        self.pipe3 = Pipe(3, architecture_config['pipes']['pipe3'], input_dim, output_dim)
        
        # Combination weights (trainable or fixed)
        self.combination_mode = 'sum'  # 'sum' or 'weighted'
        if self.combination_mode == 'weighted':
            self.combination_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x, use_routers=True):
        """
        Forward pass through all 3 pipes (parallel).
        
        Args:
            x: Input tensor [batch, input_dim] from upper model
            use_routers: If True, use sparse MicroMoE routing
        
        Returns:
            output: [batch, output_dim] for lower model
        """
        # All pipes process SAME input SIMULTANEOUSLY
        y1 = self.pipe1(x, use_router=use_routers)
        y2 = self.pipe2(x, use_router=use_routers)
        y3 = self.pipe3(x, use_router=use_routers)
        
        # Combine outputs
        if self.combination_mode == 'sum':
            output = y1 + y2 + y3
        else:
            # Weighted combination
            w = F.softmax(self.combination_weights, dim=0)
            output = w[0] * y1 + w[1] * y2 + w[2] * y3
        
        # Handle Pipe 3 padding if needed
        if 'padding_output' in self.architecture['pipes']['pipe3']:
            padding = self.architecture['pipes']['pipe3']['padding_output']
            if padding > 0:
                output = output[:, :-padding]
        
        return output
    
    def freeze_all_workers(self):
        """Freeze all workers in all pipes."""
        self.pipe1.freeze_workers()
        self.pipe2.freeze_workers()
        self.pipe3.freeze_workers()
    
    def get_trainable_parameters(self):
        """Return only trainable parameters (routers, combination weights)."""
        params = []
        params.extend(self.pipe1.router.parameters())
        params.extend(self.pipe2.router.parameters())
        params.extend(self.pipe3.router.parameters())
        if self.combination_mode == 'weighted':
            params.append(self.combination_weights)
        return params
    
    def save_architecture(self, save_dir):
        """Save complete architecture to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save full model state
        torch.save(self.state_dict(), save_path / 'full_model.pt')
        
        # Save individual pipes
        torch.save(self.pipe1.state_dict(), save_path / 'pipe1.pt')
        torch.save(self.pipe2.state_dict(), save_path / 'pipe2.pt')
        torch.save(self.pipe3.state_dict(), save_path / 'pipe3.pt')
        
        # Save architecture config
        with open(save_path / 'architecture.json', 'w') as f:
            json.dump(self.architecture, f, indent=2)
        
        print(f"✓ Saved architecture to {save_path}")
    
    def load_architecture(self, load_dir):
        """Load complete architecture from directory."""
        load_path = Path(load_dir)
        
        # Load full model state
        self.load_state_dict(torch.load(load_path / 'full_model.pt'))
        
        print(f"✓ Loaded architecture from {load_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build 3-Pipe parallel architecture with MicroMoE routers'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to architecture.json from calculate_pipes.py'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='builded_pipes',
        help='Output directory for pipe structures'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to build on'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("3-PIPE ARCHITECTURE BUILDER")
    print("=" * 70)
    print()
    
    # Load architecture configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        architecture_config = json.load(f)
    print("✓ Configuration loaded")
    print()
    
    # Display architecture info
    print("📋 Architecture Summary:")
    print(f"  Pipes: {architecture_config['num_pipes']}")
    print(f"  Workers per pipe: {architecture_config['workers_per_pipe']}")
    print(f"  Total workers: {architecture_config['total_workers']}")
    print(f"  Sparse activation: {architecture_config['sparse_activation_ratio']*100:.0f}%")
    print()
    
    print("🔄 Transformation:")
    input_dim = architecture_config['upper_model']['output_dim']
    output_dim = architecture_config['lower_model']['input_dim']
    print(f"  {input_dim:,} → {output_dim:,}")
    print()
    
    print("📐 Pipe Configurations:")
    for pipe_name, pipe_cfg in architecture_config['pipes'].items():
        dims = pipe_cfg['dimensions']
        if len(dims) == 3:
            print(f"  {pipe_name.capitalize()}: [{dims[0]}×{dims[1]}×{dims[2]}] - {pipe_cfg['structure']}")
        else:
            print(f"  {pipe_name.capitalize()}: {dims} - {pipe_cfg['structure']}")
    print()
    
    # Build architecture
    print("🏗️  Building 3-Pipe architecture...")
    device = torch.device(args.device)
    model = ThreePipeArchitecture(architecture_config).to(device)
    print("✓ Architecture built")
    print()
    
    # Display model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (routers): {trainable_params:,}")
    print(f"  Frozen (workers): {frozen_params:,}")
    print()
    
    # Display component breakdown
    print("🧩 Component Breakdown:")
    print(f"  Pipe 1:")
    print(f"    Workers: 512 × {architecture_config['pipes']['pipe1']['tile_shape']}")
    print(f"    Router: {input_dim} → 512")
    print()
    print(f"  Pipe 2:")
    print(f"    Workers: 512 × {architecture_config['pipes']['pipe2']['tile_shape']}")
    print(f"    Router: {input_dim} → 512")
    print()
    print(f"  Pipe 3:")
    print(f"    Workers: 512 × {architecture_config['pipes']['pipe3']['tile_shape']}")
    print(f"    Router: {input_dim} → 512")
    print()
    
    # Save architecture
    output_path = Path(args.output_dir)
    print(f"💾 Saving architecture to {output_path}...")
    model.save_architecture(output_path)
    print()
    
    # Save additional metadata
    metadata = {
        'model_type': '3-pipe-parallel-architecture',
        'version': '1.0',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'pipes': {
            'pipe1': {'workers': 512, 'tile_shape': architecture_config['pipes']['pipe1']['tile_shape']},
            'pipe2': {'workers': 512, 'tile_shape': architecture_config['pipes']['pipe2']['tile_shape']},
            'pipe3': {'workers': 512, 'tile_shape': architecture_config['pipes']['pipe3']['tile_shape']}
        },
        'training_notes': [
            'Workers are FROZEN - trained once via least squares',
            'Routers are TRAINABLE - learn worker selection via gradient descent',
            'Each router operates independently',
            'Sparse activation: 30% (154/512) workers per pipe'
        ]
    }
    
    with open(output_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ {output_path / 'model_metadata.json'}")
    
    # Save worker metadata for each pipe
    for pipe_id, pipe_name in [(1, 'pipe1'), (2, 'pipe2'), (3, 'pipe3')]:
        pipe_obj = getattr(model, pipe_name)
        worker_meta = pipe_obj.get_worker_metadata()
        
        with open(output_path / f'{pipe_name}_workers_metadata.json', 'w') as f:
            json.dump(worker_meta, f, indent=2)
        print(f"✓ {output_path / f'{pipe_name}_workers_metadata.json'}")
    
    print()
    print("=" * 70)
    print("✅ BUILD COMPLETE")
    print("=" * 70)
    print()
    print("📦 Built structures:")
    print(f"  - Full model: {output_path / 'full_model.pt'}")
    print(f"  - Pipe 1: {output_path / 'pipe1.pt'}")
    print(f"  - Pipe 2: {output_path / 'pipe2.pt'}")
    print(f"  - Pipe 3: {output_path / 'pipe3.pt'}")
    print()
    print("Next steps:")
    print("  1. Train workers via one-shot least squares")
    print("  2. Freeze workers")
    print("  3. (Optional) Train MicroMoE routers")
    print("  4. Deploy for inference")
    print()


if __name__ == '__main__':
    main()
