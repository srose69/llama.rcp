#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
PrisMoE: PRISM Mixture of Experts
3D Tiled Architecture with 512 Independent Workers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time
import json
import math
from pathlib import Path

NUM_WORKERS = 512

def calculate_tile_decomposition(input_dim, output_dim, num_workers=512):
    """
    Calculate optimal tile decomposition for any matrix dimensions.
    
    For 4096×4096 with 512 workers:
    - 16 row tiles × 32 col tiles = 512 workers
    - Each tile: 256 rows × 128 cols
    """
    print(f"\nTile Decomposition Calculator:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Workers: {num_workers}")
    
    aspect_ratio = output_dim / input_dim
    
    factors = []
    for i in range(1, num_workers + 1):
        if num_workers % i == 0:
            factors.append((i, num_workers // i))
    
    best_factor = min(factors, 
                      key=lambda f: abs(f[0]/f[1] - aspect_ratio))
    
    num_row_tiles, num_col_tiles = best_factor
    
    tile_rows = math.ceil(output_dim / num_row_tiles)
    tile_cols = math.ceil(input_dim / num_col_tiles)
    
    print(f"\n  Tile Grid: {num_row_tiles} rows × {num_col_tiles} cols")
    print(f"  Tile Size: {tile_rows} rows × {tile_cols} cols")
    print(f"  Total tiles: {num_row_tiles * num_col_tiles}")
    
    padded_input = num_col_tiles * tile_cols
    padded_output = num_row_tiles * tile_rows
    
    if padded_input != input_dim or padded_output != output_dim:
        print(f"\n  Padding Applied:")
        print(f"    Input: {input_dim} → {padded_input} (+{padded_input - input_dim})")
        print(f"    Output: {output_dim} → {padded_output} (+{padded_output - output_dim})")
    
    return num_row_tiles, num_col_tiles, tile_rows, tile_cols

def load_model_weights_safe(model_path):
    """Load model using transformers (safetensors/pytorch)"""
    print(f"\nLoading model: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map='cpu',
            low_cpu_mem_usage=True
        )
        lm_head = model.lm_head.weight.data
        embedding = model.model.embed_tokens.weight.data
        
        print(f"  ✓ Loaded via transformers")
        print(f"    lm_head: {lm_head.shape}")
        print(f"    embedding: {embedding.shape}")
        
        return lm_head, embedding
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None, None

def load_model_weights_gguf(model_path):
    """Load model from GGUF file"""
    print(f"\nLoading GGUF model: {model_path}")
    try:
        import gguf
        reader = gguf.GGUFReader(model_path)
        
        tensors = {}
        for tensor in reader.tensors:
            tensors[tensor.name] = torch.from_numpy(tensor.data).float()
        
        lm_head = None
        embedding = None
        
        for name in tensors.keys():
            if 'output' in name.lower() or 'lm_head' in name.lower():
                lm_head = tensors[name]
            if 'token_embd' in name.lower() or 'embed' in name.lower():
                embedding = tensors[name]
        
        if lm_head is None or embedding is None:
            print(f"  Available tensors:")
            for name in list(tensors.keys())[:20]:
                print(f"    - {name}: {tensors[name].shape}")
            
            if lm_head is None:
                lm_head = list(tensors.values())[0]
            if embedding is None:
                embedding = list(tensors.values())[1]
        
        print(f"  ✓ Loaded via gguf")
        print(f"    lm_head: {lm_head.shape}")
        print(f"    embedding: {embedding.shape}")
        
        return lm_head, embedding
        
    except ImportError:
        print(f"  ✗ gguf library not installed. Install: pip install gguf")
        return None, None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None, None

def load_model_weights(model_path):
    """Universal model loader"""
    if str(model_path).endswith('.gguf'):
        return load_model_weights_gguf(model_path)
    else:
        return load_model_weights_safe(model_path)

class PrisMoEWorker:
    """Single independent worker handling one tile"""
    def __init__(self, worker_id, tile_rows, tile_cols, row_idx, col_idx):
        self.worker_id = worker_id
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.weight = None
    
    def get_input_range(self, input_dim):
        """Calculate input slice for this worker"""
        start = self.col_idx * self.tile_cols
        end = min(start + self.tile_cols, input_dim)
        return start, end
    
    def get_output_range(self, output_dim):
        """Calculate output slice for this worker"""
        start = self.row_idx * self.tile_rows
        end = min(start + self.tile_rows, output_dim)
        return start, end
    
    def train_with_residual(self, X_full, residual_full):
        """
        Train this tile given residual from other tiles.
        Uses Alternating Least Squares approach.
        
        X_full: [batch, input_dim] - full input
        residual_full: [batch, output_dim] - target residual for this tile's output region
        """
        input_dim = X_full.shape[1]
        output_dim = residual_full.shape[1]
        
        in_start, in_end = self.get_input_range(input_dim)
        out_start, out_end = self.get_output_range(output_dim)
        
        X_chunk = X_full[:, in_start:in_end]
        residual_chunk = residual_full[:, out_start:out_end]
        
        actual_in = in_end - in_start
        actual_out = out_end - out_start
        
        if actual_in < self.tile_cols or actual_out < self.tile_rows:
            X_padded = torch.zeros(X_chunk.shape[0], self.tile_cols, device=X_chunk.device)
            residual_padded = torch.zeros(residual_chunk.shape[0], self.tile_rows, device=residual_chunk.device)
            X_padded[:, :actual_in] = X_chunk
            residual_padded[:, :actual_out] = residual_chunk
            X_chunk = X_padded
            residual_chunk = residual_padded
        
        self.weight = residual_chunk.T @ torch.linalg.pinv(X_chunk).T
        
        prediction = X_chunk @ self.weight.T
        error = F.mse_loss(prediction[:, :actual_out], residual_full[:, out_start:out_end]).item()
        return error
    
    def forward(self, X_full):
        """
        Forward pass for this worker
        X_full: [batch, input_dim]
        Returns: [batch, tile_rows]
        """
        in_start, in_end = self.get_input_range(X_full.shape[1])
        X_chunk = X_full[:, in_start:in_end]
        
        actual_in = in_end - in_start
        if actual_in < self.tile_cols:
            X_padded = torch.zeros(X_chunk.shape[0], self.tile_cols, device=X_chunk.device)
            X_padded[:, :actual_in] = X_chunk
            X_chunk = X_padded
        
        return X_chunk @ self.weight.T

class PrisMoE(nn.Module):
    """
    PrisMoE: Mixture of Expert Workers
    NO matrix assembly - direct pipeline architecture
    """
    def __init__(self, input_dim, output_dim, num_workers=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_workers = num_workers
        
        num_row_tiles, num_col_tiles, tile_rows, tile_cols = calculate_tile_decomposition(
            input_dim, output_dim, num_workers
        )
        
        self.num_row_tiles = num_row_tiles
        self.num_col_tiles = num_col_tiles
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        
        self.workers = []
        for worker_id in range(num_workers):
            row_idx = worker_id // num_col_tiles
            col_idx = worker_id % num_col_tiles
            
            worker = PrisMoEWorker(
                worker_id, tile_rows, tile_cols, row_idx, col_idx
            )
            self.workers.append(worker)
        
        print(f"\n✓ PrisMoE initialized:")
        print(f"  {num_workers} workers")
        print(f"  Grid: {num_row_tiles}×{num_col_tiles}")
        print(f"  Tile: {tile_rows}×{tile_cols}")
    
    def train_oneshot(self, X, Z, max_iters=50, tolerance=1e-6):
        """
        Train all 512 workers using Alternating Least Squares.
        
        Key insight: Each output row Z[row_i] is sum of contributions from 
        all 32 column tiles in that row group.
        
        X: [batch, input_dim]
        Z: [batch, output_dim]
        max_iters: maximum ALS iterations
        tolerance: convergence threshold
        """
        print(f"\nTraining {self.num_workers} workers with Alternating Least Squares...")
        print(f"  Max iterations: {max_iters}")
        print(f"  Tolerance: {tolerance}")
        
        t0 = time.time()
        
        for worker in self.workers:
            worker.weight = torch.randn(worker.tile_rows, worker.tile_cols, device=X.device) * 0.01
        
        prev_loss = float('inf')
        
        for iteration in range(max_iters):
            for i, worker in enumerate(self.workers):
                out_start, out_end = worker.get_output_range(self.output_dim)
                
                contribution_from_others = torch.zeros(X.shape[0], out_end - out_start, device=X.device)
                
                for j, other_worker in enumerate(self.workers):
                    if other_worker.row_idx == worker.row_idx and j != i:
                        other_out = other_worker.forward(X)
                        contribution_from_others += other_out[:, :out_end - out_start]
                
                target_for_this_worker = Z[:, out_start:out_end] - contribution_from_others
                
                residual_full = torch.zeros_like(Z)
                residual_full[:, out_start:out_end] = target_for_this_worker
                
                error = worker.train_with_residual(X, residual_full)
            
            current_output = self.forward(X)
            current_loss = F.mse_loss(current_output, Z).item()
            
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"  Iteration {iteration+1}/{max_iters}: loss={current_loss:.10f}")
            
            if abs(prev_loss - current_loss) < tolerance:
                print(f"  ✓ Converged at iteration {iteration+1}")
                break
            
            prev_loss = current_loss
        
        train_time = (time.time() - t0) * 1000
        final_output = self.forward(X)
        final_loss = F.mse_loss(final_output, Z).item()
        
        print(f"\n✓ Training complete:")
        print(f"  Time: {train_time:.2f}ms")
        print(f"  Iterations: {iteration+1}")
        print(f"  Final loss: {final_loss:.10f}")
        
        return train_time, [final_loss] * self.num_workers
    
    def forward(self, X):
        """
        Inference via direct pipeline routing
        NO assembly into full matrix!
        
        X: [batch, input_dim]
        Returns: [batch, output_dim]
        """
        batch_size = X.shape[0]
        output = torch.zeros(batch_size, self.output_dim, device=X.device)
        
        for worker in self.workers:
            out_start, out_end = worker.get_output_range(self.output_dim)
            worker_output = worker.forward(X)
            
            actual_out = out_end - out_start
            output[:, out_start:out_end] = worker_output[:, :actual_out]
        
        return output
    
    def get_weights_3d(self):
        """Export weights as 3D tensor for analysis"""
        weights = torch.stack([w.weight for w in self.workers])
        return weights
    
    def count_parameters(self):
        """Count total parameters across all workers"""
        return sum(w.weight.numel() for w in self.workers)

class FullMatrixGD(nn.Module):
    """Baseline: standard full matrix"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
    
    def forward(self, x):
        return x @ self.weight.T

def train_gd_baseline(X, Z, input_dim, output_dim, target_error=1e-9, max_iters=10000):
    """GD baseline for comparison"""
    print("\n" + "="*80)
    print("GRADIENT DESCENT BASELINE")
    print("="*80)
    print(f"  Target error: {target_error}")
    print(f"  Max iterations: {max_iters}")
    
    dev = X.device
    model = FullMatrixGD(input_dim, output_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    t0 = time.time()
    loss_history = []
    
    for i in range(max_iters):
        opt.zero_grad()
        loss = F.mse_loss(model(X), Z)
        loss.backward()
        opt.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if (i + 1) % 100 == 0:
            print(f"    Iter {i+1}: {loss_val:.10f}")
        
        if loss_val < target_error:
            print(f"  ✓ Converged at iteration {i+1}")
            break
    
    gd_time = (time.time() - t0) * 1000
    final_loss = F.mse_loss(model(X), Z).item()
    
    print(f"\n  ✓ Final loss: {final_loss:.10f}")
    print(f"  ✓ Time: {gd_time:.2f}ms")
    print(f"  ✓ Iterations: {i+1}")
    
    return model, final_loss, gd_time, i+1, loss_history

def run_benchmark(model_path_A, model_path_B, batch_size=1000):
    """Main benchmark"""
    print("="*80)
    print("PrisMoE: PRISM Mixture of Experts")
    print("3D Tiled Architecture Benchmark")
    print("="*80)
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {dev}")
    
    print(f"\nModel A (source): {model_path_A}")
    print(f"Model B (target): {model_path_B}")
    
    lm_A, emb_A = load_model_weights(model_path_A)
    lm_B, emb_B = load_model_weights(model_path_B)
    
    if lm_A is None or lm_B is None:
        print("\n✗ Failed to load models!")
        return
    
    input_dim = lm_A.shape[1] if len(lm_A.shape) > 1 else lm_A.shape[0]
    output_dim = emb_B.shape[1] if len(emb_B.shape) > 1 else emb_B.shape[0]
    
    print(f"\n" + "="*80)
    print("ARCHITECTURE")
    print("="*80)
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Batch size: {batch_size}")
    
    print(f"\n" + "="*80)
    print("DATA GENERATION")
    print("="*80)
    X = torch.randn(batch_size, input_dim, device=dev)
    Z = torch.randn(batch_size, output_dim, device=dev)
    print(f"  X: {X.shape}")
    print(f"  Z: {Z.shape}")
    
    print(f"\n" + "="*80)
    print("PRISMOE: TILED ARCHITECTURE")
    print("="*80)
    prismoe = PrisMoE(input_dim, output_dim, NUM_WORKERS)
    train_time, worker_errors = prismoe.train_oneshot(X, Z)
    
    with torch.no_grad():
        prismoe_output = prismoe(X)
    prismoe_error = F.mse_loss(prismoe_output, Z).item()
    
    print(f"\nPrisMoE Results:")
    print(f"  Final error: {prismoe_error:.10f}")
    print(f"  Training time: {train_time:.2f}ms")
    print(f"  Parameters: {prismoe.count_parameters():,}")
    
    gd_model, gd_error, gd_time, gd_iters, loss_history = train_gd_baseline(
        X, Z, input_dim, output_dim
    )
    
    with torch.no_grad():
        gd_output = gd_model(X)
    
    print(f"\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    weights_3d = prismoe.get_weights_3d()
    torch.save(weights_3d.cpu(), 'prismoe_weights.pt')
    torch.save(gd_model.weight.data.cpu(), 'prismoe_gd_weight.pt')
    torch.save(prismoe_output.cpu(), 'prismoe_output.pt')
    torch.save(gd_output.cpu(), 'prismoe_gd_output.pt')
    torch.save(Z.cpu(), 'prismoe_target.pt')
    torch.save(worker_errors, 'prismoe_worker_errors.pt')
    torch.save(loss_history, 'prismoe_gd_loss_history.pt')
    print("  ✓ Saved all outputs")
    
    print(f"\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    output_diff = prismoe_output - gd_output
    mse_diff = F.mse_loss(prismoe_output, gd_output).item()
    cos_sim = F.cosine_similarity(
        prismoe_output.flatten().unsqueeze(0),
        gd_output.flatten().unsqueeze(0)
    ).item()
    
    print(f"\nAccuracy:")
    print(f"  PrisMoE error: {prismoe_error:.10f}")
    print(f"  GD error: {gd_error:.10f}")
    print(f"  Accuracy ratio: {prismoe_error/gd_error:.4f}×")
    
    print(f"\nPerformance:")
    print(f"  PrisMoE time: {train_time:.2f}ms")
    print(f"  GD time: {gd_time:.2f}ms ({gd_iters} iters)")
    print(f"  Speedup: {gd_time/train_time:.2f}×")
    
    print(f"\nOutput Similarity:")
    print(f"  MSE difference: {mse_diff:.10f}")
    print(f"  Cosine similarity: {cos_sim:.10f}")
    print(f"  Max abs diff: {output_diff.abs().max().item():.10f}")
    
    print(f"\nArchitecture Analysis:")
    print(f"  PrisMoE workers: {prismoe.num_workers}")
    print(f"  Tile grid: {prismoe.num_row_tiles}×{prismoe.num_col_tiles}")
    print(f"  Tile size: {prismoe.tile_rows}×{prismoe.tile_cols}")
    print(f"  PrisMoE params: {prismoe.count_parameters():,}")
    print(f"  GD params: {gd_model.weight.numel():,}")
    print(f"  Param ratio: {prismoe.count_parameters()/gd_model.weight.numel():.2f}×")
    
    results = {
        'architecture': {
            'model_A': str(model_path_A),
            'model_B': str(model_path_B),
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_workers': NUM_WORKERS,
            'tile_grid': [prismoe.num_row_tiles, prismoe.num_col_tiles],
            'tile_size': [prismoe.tile_rows, prismoe.tile_cols],
            'batch_size': batch_size
        },
        'prismoe': {
            'method': 'Tiled one-shot LS with 512 workers',
            'error': prismoe_error,
            'time_ms': train_time,
            'params': prismoe.count_parameters(),
            'avg_worker_error': sum(worker_errors) / len(worker_errors)
        },
        'gd_baseline': {
            'method': 'Full matrix gradient descent',
            'error': gd_error,
            'time_ms': gd_time,
            'iterations': gd_iters,
            'params': int(gd_model.weight.numel())
        },
        'comparison': {
            'speedup': gd_time / train_time,
            'accuracy_ratio': prismoe_error / gd_error,
            'output_mse_diff': mse_diff,
            'output_cosine_sim': cos_sim,
            'max_abs_diff': float(output_diff.abs().max().item())
        }
    }
    
    with open('output/prismoe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: output/prismoe_results.json")

def main():
    parser = argparse.ArgumentParser(description='PrisMoE Benchmark')
    parser.add_argument('model_A', type=str, help='Source model path')
    parser.add_argument('model_B', type=str, help='Target model path (or same as A)')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size')
    
    args = parser.parse_args()
    
    run_benchmark(args.model_A, args.model_B, args.batch)

if __name__ == "__main__":
    main()
