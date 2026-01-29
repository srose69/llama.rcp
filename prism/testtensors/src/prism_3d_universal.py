#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
PRISM 3D Universal Adapter
Creates 3D tensor decomposition between any two models
Uses 512 parallel micro-layers solved via one-shot least squares
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import argparse
import time
import json
from pathlib import Path

NUM_MICRO_LAYERS = 512

def load_model_weights(model_path):
    """Load lm_head and embedding from model"""
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    lm_head = model.lm_head.weight.data
    embedding = model.model.embed_tokens.weight.data
    print(f"  lm_head: {lm_head.shape}")
    print(f"  embedding: {embedding.shape}")
    return lm_head, embedding

def create_3d_decomposition(input_dim, output_dim, num_micro=512):
    """Calculate 3D tensor decomposition architecture"""
    print(f"\n3D Tensor Decomposition:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Micro-layers: {num_micro}")
    
    if input_dim % num_micro != 0:
        raise ValueError(f"Input dim {input_dim} must be divisible by {num_micro}")
    if output_dim % num_micro != 0:
        raise ValueError(f"Output dim {output_dim} must be divisible by {num_micro}")
    
    micro_in = input_dim // num_micro
    micro_out = output_dim // num_micro
    
    print(f"  Each micro-layer: {micro_in} → {micro_out}")
    print(f"  Total params: {num_micro * micro_in * micro_out:,}")
    print(f"  vs Full matrix: {input_dim * output_dim:,}")
    ratio = (input_dim * output_dim) / (num_micro * micro_in * micro_out)
    print(f"  Compression: {ratio:.2f}× (should be ~1.0 for same dims)")
    
    return micro_in, micro_out

def solve_3d_oneshot(X, Z, num_micro=512):
    """
    Solve 512 independent least-squares problems
    X: [batch, input_dim]
    Z: [batch, output_dim]
    Returns: weights [num_micro, micro_out, micro_in]
    """
    batch, input_dim = X.shape
    output_dim = Z.shape[1]
    
    micro_in = input_dim // num_micro
    micro_out = output_dim // num_micro
    
    print("\nSolving 3D One-Shot Least Squares...")
    print(f"  Splitting into {num_micro} independent problems")
    
    weights = []
    t0 = time.time()
    
    for i in range(num_micro):
        in_start = i * micro_in
        in_end = (i + 1) * micro_in
        out_start = i * micro_out
        out_end = (i + 1) * micro_out
        
        X_micro = X[:, in_start:in_end]
        Z_micro = Z[:, out_start:out_end]
        
        W_micro = Z_micro.T @ torch.linalg.pinv(X_micro).T
        weights.append(W_micro)
        
        if (i + 1) % 100 == 0:
            err = F.mse_loss(X_micro @ W_micro.T, Z_micro).item()
            print(f"    Micro-layer {i+1}/{num_micro}: error={err:.10f}")
    
    weights_tensor = torch.stack(weights)
    solve_time = (time.time() - t0) * 1000
    
    print(f"  ✓ Solved {num_micro} micro-layers in {solve_time:.2f}ms")
    return weights_tensor, solve_time

def forward_3d(X, weights_3d):
    """Forward pass through 3D tensor"""
    batch, input_dim = X.shape
    num_micro, micro_out, micro_in = weights_3d.shape
    output_dim = num_micro * micro_out
    
    outputs = []
    for i in range(num_micro):
        in_start = i * micro_in
        in_end = (i + 1) * micro_in
        X_micro = X[:, in_start:in_end]
        out_micro = X_micro @ weights_3d[i].T
        outputs.append(out_micro)
    
    return torch.cat(outputs, dim=1)

def evaluate_3d(X, Z, weights_3d):
    """Evaluate 3D tensor reconstruction"""
    output = forward_3d(X, weights_3d)
    error = F.mse_loss(output, Z).item()
    return output, error

class FullMatrixGD(nn.Module):
    """Standard full matrix for GD comparison"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
    
    def forward(self, x):
        return x @ self.weight.T

def train_gd_baseline(X, Z, input_dim, output_dim, target_error=1e-9, max_iters=10000):
    """Train full matrix GD baseline"""
    print("\nGradient Descent Baseline:")
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
    
    print(f"  ✓ Final loss: {final_loss:.10f}")
    print(f"  ✓ Time: {gd_time:.2f}ms")
    print(f"  ✓ Iterations: {i+1}")
    
    return model.weight.data, final_loss, gd_time, i+1, loss_history

def run_benchmark(input_model_path, output_model_path, batch_size=1000):
    """Main benchmark function"""
    print("="*80)
    print("PRISM 3D UNIVERSAL ADAPTER")
    print("="*80)
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}\n")
    
    print(f"Input model: {input_model_path}")
    print(f"Output model: {output_model_path}")
    print(f"Batch size: {batch_size}\n")
    
    lm_head_in, emb_in = load_model_weights(input_model_path)
    lm_head_out, emb_out = load_model_weights(output_model_path)
    
    input_dim = lm_head_in.shape[1]
    output_dim = emb_out.shape[1]
    
    print(f"\nAdapter Configuration:")
    print(f"  Source: {Path(input_model_path).name} lm_head ({input_dim} dim)")
    print(f"  Target: {Path(output_model_path).name} embedding ({output_dim} dim)")
    
    micro_in, micro_out = create_3d_decomposition(input_dim, output_dim, NUM_MICRO_LAYERS)
    
    print("\n" + "="*80)
    print("GENERATING DATA")
    print("="*80)
    X = torch.randn(batch_size, input_dim, device=dev)
    Z = torch.randn(batch_size, output_dim, device=dev)
    print(f"  X: {X.shape}")
    print(f"  Z: {Z.shape}")
    
    print("\n" + "="*80)
    print("ADVANCED SOLVER: 3D ONE-SHOT LEAST SQUARES")
    print("="*80)
    weights_3d, solve_time = solve_3d_oneshot(X, Z, NUM_MICRO_LAYERS)
    adv_output, adv_error = evaluate_3d(X, Z, weights_3d)
    
    print(f"\nAdvanced Solver Results:")
    print(f"  Error: {adv_error:.10f}")
    print(f"  Time: {solve_time:.2f}ms")
    print(f"  Weights shape: {weights_3d.shape}")
    
    print("\n" + "="*80)
    print("GRADIENT DESCENT BASELINE")
    print("="*80)
    gd_weight, gd_error, gd_time, gd_iters, loss_history = train_gd_baseline(
        X, Z, input_dim, output_dim, target_error=1e-9, max_iters=10000
    )
    gd_output = X @ gd_weight.T
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    torch.save(weights_3d.cpu(), 'prism_3d_weights.pt')
    torch.save(gd_weight.cpu(), 'prism_gd_weight.pt')
    torch.save(adv_output.cpu(), 'prism_3d_output.pt')
    torch.save(gd_output.cpu(), 'prism_gd_output.pt')
    torch.save(Z.cpu(), 'prism_target.pt')
    torch.save(loss_history, 'prism_gd_loss_history.pt')
    print("  ✓ Saved all outputs and weights")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    diff = adv_output - gd_output
    mse_diff = F.mse_loss(adv_output, gd_output).item()
    cos_sim = F.cosine_similarity(
        adv_output.flatten().unsqueeze(0),
        gd_output.flatten().unsqueeze(0)
    ).item()
    
    print(f"\nOutput Comparison:")
    print(f"  Advanced error: {adv_error:.10f}")
    print(f"  GD error: {gd_error:.10f}")
    print(f"  Output MSE diff: {mse_diff:.10f}")
    print(f"  Cosine similarity: {cos_sim:.10f}")
    
    print(f"\nPerformance:")
    print(f"  Advanced time: {solve_time:.2f}ms (0 iterations)")
    print(f"  GD time: {gd_time:.2f}ms ({gd_iters} iterations)")
    print(f"  Speedup: {gd_time/solve_time:.2f}×")
    
    accuracy_ratio = adv_error / gd_error
    print(f"\nAccuracy:")
    print(f"  Ratio (Adv/GD): {accuracy_ratio:.4f}×")
    if adv_error < gd_error:
        improvement = (1 - accuracy_ratio) * 100
        print(f"  ✅ Advanced is {improvement:.1f}% more accurate")
    else:
        degradation = (accuracy_ratio - 1) * 100
        print(f"  ⚠️ Advanced is {degradation:.1f}% less accurate")
    
    print(f"\n3D Tensor Analysis:")
    print(f"  Micro-layers: {NUM_MICRO_LAYERS}")
    print(f"  Each micro-layer: {micro_in}×{micro_out}")
    print(f"  Total 3D params: {weights_3d.numel():,}")
    print(f"  Full matrix params: {gd_weight.numel():,}")
    print(f"  Param ratio: {weights_3d.numel() / gd_weight.numel():.2f}×")
    
    results = {
        'architecture': {
            'input_model': str(input_model_path),
            'output_model': str(output_model_path),
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_micro_layers': NUM_MICRO_LAYERS,
            'micro_in': micro_in,
            'micro_out': micro_out,
            'batch_size': batch_size
        },
        'advanced_solver': {
            'method': '3D one-shot least squares',
            'error': adv_error,
            'time_ms': solve_time,
            'iterations': 0,
            'weights_shape': list(weights_3d.shape)
        },
        'gd_baseline': {
            'method': 'Full matrix gradient descent',
            'error': gd_error,
            'time_ms': gd_time,
            'iterations': gd_iters,
            'weights_shape': list(gd_weight.shape)
        },
        'comparison': {
            'speedup': gd_time / solve_time,
            'accuracy_ratio': accuracy_ratio,
            'improvement_pct': (1 - accuracy_ratio) * 100,
            'output_mse_diff': mse_diff,
            'output_cosine_sim': cos_sim,
            'param_ratio': weights_3d.numel() / gd_weight.numel()
        }
    }
    
    with open('prism_3d_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved: prism_3d_results.json")

def main():
    parser = argparse.ArgumentParser(description='PRISM 3D Universal Adapter')
    parser.add_argument('input_model', type=str, help='Path to input model')
    parser.add_argument('output_model', type=str, help='Path to output model')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size (default: 1000)')
    
    args = parser.parse_args()
    
    run_benchmark(args.input_model, args.output_model, args.batch)

if __name__ == "__main__":
    main()
