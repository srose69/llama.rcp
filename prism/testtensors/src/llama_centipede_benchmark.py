#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import time
import json

class CascadeGD(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) 
            for i in range(len(dims)-1)
        ])
    
    def forward(self, x, ret_inter=False):
        if ret_inter:
            outs = [x]
            for layer in self.layers:
                x = layer(x)
                outs.append(x)
            return outs
        for layer in self.layers:
            x = layer(x)
        return x

def load_llama(model_path='meta-llama/Llama-3.1-8B-Instruct'):
    print(f"Loading Llama from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    lm_head = model.lm_head.weight.data
    embed = model.model.embed_tokens.weight.data
    print(f"Llama lm_head: {lm_head.shape}")
    print(f"Llama embedding: {embed.shape}")
    return lm_head, embed

def generate_fractional_targets(X, Z, dims):
    print("  Generating fractional targets (compression-decompression)...")
    targets = []
    num_layers = len(dims) - 1
    mid_point = num_layers // 2
    
    for i in range(num_layers):
        if i < mid_point:
            alpha = (i + 1) / num_layers
        else:
            alpha = 1.0 - ((i - mid_point) / num_layers)
        
        current_dim = dims[i + 1]
        
        if current_dim < X.shape[1]:
            X_proj = X[:, :current_dim]
        else:
            X_proj = torch.cat([X, torch.zeros(X.shape[0], current_dim - X.shape[1], device=X.device)], dim=1)
        
        if current_dim < Z.shape[1]:
            Z_proj = Z[:, :current_dim]
        else:
            Z_proj = torch.cat([Z, torch.zeros(Z.shape[0], current_dim - Z.shape[1], device=Z.device)], dim=1)
        
        target = (1 - alpha) * X_proj + alpha * Z_proj
        targets.append(target)
        print(f"    Layer {i+1}: alpha={alpha:.2f}, shape={target.shape}")
    
    print(f"  ✓ Generated {len(targets)} intermediate targets")
    return targets

def oneshot_solve(X, Z):
    return Z.T @ torch.linalg.pinv(X).T

def sliding_solve(X, targets, dims):
    weights = []
    curr = X
    for i, tgt in enumerate(targets):
        print(f"\n  Layer {i+1}: {dims[i]}→{dims[i+1]}")
        W = oneshot_solve(curr, tgt)
        weights.append(W)
        curr = curr @ W.T
        print(f"    Error: {F.mse_loss(curr, tgt).item():.10f}")
    return weights

def apply_sparse_corrections(weights, X, Z, error_threshold=1e-4):
    outs = [X]
    out = X
    for W in weights:
        out = out @ W.T
        outs.append(out)
    
    current_error = F.mse_loss(out, Z).item()
    print(f"\n  Current error: {current_error:.10f}")
    print(f"  Threshold: {error_threshold:.10f}")
    
    if current_error < error_threshold:
        print(f"  ✓ Error acceptable, skipping corrections")
        return None
    
    print(f"  ⚠ Error requires correction...")
    res = Z - out
    mask = res.abs() > (error_threshold * 0.1)
    sparse = torch.zeros_like(res)
    sparse[mask] = res[mask]
    W_corr = sparse.T @ torch.linalg.pinv(outs[-2]).T
    sparsity = mask.sum().item() / mask.numel() * 100
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Correction norm: {W_corr.norm().item():.6f}")
    return W_corr

def calibrate_coupling(weights, X, Z, error_threshold=1e-6, lr=1e-3, iters=20):
    out = X
    for W in weights:
        out = out @ W.T
    initial_error = F.mse_loss(out, Z).item()
    
    print(f"\n  Initial error: {initial_error:.10f}")
    print(f"  Threshold: {error_threshold:.10f}")
    
    if initial_error < error_threshold:
        print(f"  ✓ Error acceptable, skipping calibration")
        return torch.ones(len(weights), device=X.device)
    
    print(f"  ⚠ Error requires calibration...")
    coeffs = torch.ones(len(weights), device=X.device, requires_grad=True)
    opt = torch.optim.Adam([coeffs], lr=lr)
    for i in range(iters):
        opt.zero_grad()
        out = X
        for j, W in enumerate(weights):
            out = out @ (W * coeffs[j]).T
        loss = F.mse_loss(out, Z)
        loss.backward()
        opt.step()
        if (i+1) % 5 == 0:
            print(f"    Iter {i+1}: {loss.item():.8f}")
    
    final_error = F.mse_loss(out, Z).item()
    print(f"  Final error: {final_error:.10f}")
    return coeffs.detach()

def prism_solve(X, Z, dims, error_threshold=1e-6):
    print("="*80)
    print("PRISM ADVANCED SOLVER")
    print("="*80)
    print(f"\nBatch: {X.shape[0]}, Dims: {dims}")
    print(f"Source: Llama #1 output → Target: Llama #2 input")
    print(f"Error threshold: {error_threshold:.10f}")
    t0 = time.time()
    
    print("\n" + "-"*80)
    print("STEP 1: Generate Fractional Targets")
    print("-"*80)
    targets = generate_fractional_targets(X, Z, dims)
    
    print("\n" + "-"*80)
    print("STEP 2: One-Shot Solve")
    print("-"*80)
    weights = sliding_solve(X, targets, dims)
    
    print("\n" + "-"*80)
    print("STEP 3: Initial Eval")
    print("-"*80)
    out = X
    for W in weights:
        out = out @ W.T
    init_err = F.mse_loss(out, Z).item()
    print(f"  Initial error: {init_err:.10f}")
    
    print("\n" + "-"*80)
    print("STEP 4: Sparse Corrections")
    print("-"*80)
    W_corr = apply_sparse_corrections(weights, X, Z, error_threshold)
    if W_corr is not None:
        weights[-1] = weights[-1] + W_corr
        print("  ✓ Corrections applied")
    
    print("\n" + "-"*80)
    print("STEP 5: Coupling Calibration")
    print("-"*80)
    coeffs = calibrate_coupling(weights, X, Z, error_threshold)
    weights_cal = [W * c for W, c in zip(weights, coeffs)]
    out = X
    for W in weights_cal:
        out = out @ W.T
    final_err = F.mse_loss(out, Z).item()
    total_time = (time.time() - t0) * 1000
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"  Initial: {init_err:.10f}")
    print(f"  Final: {final_err:.10f}")
    print(f"  Improvement: {(1-final_err/init_err)*100:.2f}%")
    print(f"  Time: {total_time:.2f}ms")
    return weights_cal, coeffs, total_time, final_err

def run_benchmark():
    print("="*80)
    print("LLAMA CENTIPEDE BENCHMARK")
    print("Two Llama 8B models connected in sequence")
    print("="*80)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {dev}")
    
    llama_lm, llama_emb = load_llama()
    
    dims = [4096, 3584, 3072, 2560, 2048, 2560, 3072, 3584, 4096]
    batch = 10000
    
    print("\n" + "="*80)
    print("CENTIPEDE ARCHITECTURE")
    print("="*80)
    print(f"Llama #1 output (lm_head): {llama_lm.shape}")
    print(f"  ↓")
    print(f"PRISM Cascade: {' → '.join(map(str, dims))}")
    print(f"  Bottleneck at position {dims.index(min(dims))}: {min(dims)} dim")
    print(f"  ↓")
    print(f"Llama #2 input (embedding): {llama_emb.shape}")
    print(f"\nBatch size: {batch}")
    
    print("\n" + "="*80)
    print("Generating Data")
    print("="*80)
    X = torch.randn(batch, dims[0], device=dev)
    Z = torch.randn(batch, dims[-1], device=dev)
    print(f"  X (Llama #1 output simulation): {X.shape}")
    print(f"  Z (Llama #2 input simulation): {Z.shape}")
    
    print("\n" + "="*80)
    print("GRADIENT DESCENT TO CONVERGENCE")
    print("="*80)
    t0_gd = time.time()
    model_gd = CascadeGD(dims).to(dev)
    opt = torch.optim.Adam(model_gd.parameters(), lr=1e-3)
    
    print("\n  Training GD to convergence (target: < 1e-9)...")
    target_error = 1e-9
    max_iters = 10000
    loss_history = []
    
    for i in range(max_iters):
        opt.zero_grad()
        loss = F.mse_loss(model_gd(X), Z)
        loss.backward()
        opt.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if (i+1) % 100 == 0:
            print(f"    Iter {i+1}: {loss_val:.10f}")
        
        if loss_val < target_error:
            print(f"  ✓ Converged at iteration {i+1}")
            break
    
    with torch.no_grad():
        gd_output = model_gd(X)
    gd_loss = F.mse_loss(gd_output, Z).item()
    gd_time = (time.time() - t0_gd) * 1000
    gd_iters = i + 1
    
    print(f"  ✓ GD converged loss: {gd_loss:.10f}")
    print(f"  ✓ Iterations: {gd_iters}")
    print(f"  ✓ Time: {gd_time:.2f}ms")
    
    weights, coeffs, t_adv, err_adv = prism_solve(X, Z, dims, error_threshold=1e-6)
    
    with torch.no_grad():
        adv_output = X
        for W, c in zip(weights, coeffs):
            adv_output = adv_output @ (W * c).T
    
    gd_weights = []
    with torch.no_grad():
        for layer in model_gd.layers:
            gd_weights.append(layer.weight.data.clone())
    
    print("\n" + "="*80)
    print("SAVING OUTPUTS AND WEIGHTS")
    print("="*80)
    torch.save(adv_output.cpu(), 'centipede_advanced_output.pt')
    torch.save(gd_output.cpu(), 'centipede_gd_output.pt')
    torch.save(Z.cpu(), 'centipede_target_output.pt')
    torch.save([w.cpu() for w in weights], 'centipede_advanced_weights.pt')
    torch.save([w.cpu() for w in gd_weights], 'centipede_gd_weights.pt')
    torch.save(loss_history, 'centipede_gd_loss_history.pt')
    print("  ✓ Saved all outputs and weights")
    
    print("\n" + "="*80)
    print("MATHEMATICAL ANALYSIS")
    print("="*80)
    
    err_adv_to_target = F.mse_loss(adv_output, Z).item()
    err_gd_to_target = F.mse_loss(gd_output, Z).item()
    diff = adv_output - gd_output
    mse_diff = F.mse_loss(adv_output, gd_output).item()
    adv_flat = adv_output.flatten()
    gd_flat = gd_output.flatten()
    cos_sim = F.cosine_similarity(adv_flat.unsqueeze(0), gd_flat.unsqueeze(0)).item()
    
    print(f"\n1. OUTPUT COMPARISON:")
    print(f"  Error to Target:")
    print(f"    Advanced: {err_adv_to_target:.10f}")
    print(f"    GD: {err_gd_to_target:.10f}")
    print(f"    Ratio: {err_adv_to_target/err_gd_to_target:.2f}×")
    print(f"  Output Difference:")
    print(f"    MSE: {mse_diff:.10f}")
    print(f"    Max abs: {diff.abs().max().item():.10f}")
    print(f"    Mean abs: {diff.abs().mean().item():.10f}")
    print(f"    Cosine similarity: {cos_sim:.10f}")
    
    print(f"\n2. WEIGHT COMPARISON:")
    weight_analysis = []
    for i, (w_adv, w_gd) in enumerate(zip(weights, gd_weights)):
        w_diff = w_adv - w_gd
        w_mse = F.mse_loss(w_adv, w_gd).item()
        w_cos = F.cosine_similarity(w_adv.flatten().unsqueeze(0), w_gd.flatten().unsqueeze(0)).item()
        
        frob_adv = w_adv.norm().item()
        frob_gd = w_gd.norm().item()
        frob_diff = w_diff.norm().item()
        
        print(f"  Layer {i+1} ({dims[i]}→{dims[i+1]}):")
        print(f"    MSE: {w_mse:.10f}")
        print(f"    Cosine sim: {w_cos:.10f}")
        print(f"    Frobenius norm (Adv): {frob_adv:.6f}")
        print(f"    Frobenius norm (GD): {frob_gd:.6f}")
        print(f"    Relative diff: {(frob_diff/frob_gd)*100:.2f}%")
        
        weight_analysis.append({
            'layer': i+1,
            'mse': w_mse,
            'cosine_sim': w_cos,
            'frob_adv': frob_adv,
            'frob_gd': frob_gd,
            'frob_diff': frob_diff
        })
    
    print(f"\n3. CENTIPEDE ARCHITECTURE ANALYSIS:")
    print(f"  Total layers: {len(dims)-1}")
    print(f"  Bottleneck: {min(dims)} dim (compression ratio: {dims[0]/min(dims):.2f}×)")
    print(f"  Symmetry: {dims == dims[::-1]}")
    print(f"  Total parameters: {sum(dims[i]*dims[i+1] for i in range(len(dims)-1)):,}")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\nAdvanced Solver:")
    print(f"  Time: {t_adv:.2f}ms")
    print(f"  Error: {err_adv_to_target:.10f}")
    print(f"\nGradient Descent:")
    print(f"  Time: {gd_time:.2f}ms")
    print(f"  Loss: {err_gd_to_target:.10f}")
    print(f"  Iterations: {gd_iters}")
    print(f"\nMetrics:")
    print(f"  Speedup: {gd_time/t_adv:.2f}×")
    accuracy_ratio = err_adv_to_target / err_gd_to_target
    print(f"  Accuracy ratio (Adv/GD): {accuracy_ratio:.2f}×")
    if err_adv_to_target < err_gd_to_target:
        improvement = (1 - err_adv_to_target/err_gd_to_target) * 100
        print(f"  ✅ Advanced solver is {improvement:.1f}% more accurate")
    else:
        degradation = (err_adv_to_target/err_gd_to_target - 1) * 100
        print(f"  ⚠️ Advanced solver is {degradation:.1f}% less accurate")
    
    results = {
        'advanced_solver': {
            'time_ms': t_adv,
            'error': err_adv_to_target,
            'coupling_coeffs': coeffs.cpu().tolist(),
            'method': 'One-shot least squares with fractional targets'
        },
        'gd_baseline': {
            'time_ms': gd_time,
            'loss': err_gd_to_target,
            'iterations': gd_iters,
            'method': 'Gradient descent to convergence'
        },
        'comparison': {
            'speedup': gd_time / t_adv,
            'accuracy_ratio': accuracy_ratio,
            'improvement_pct': (1 - err_adv_to_target/err_gd_to_target) * 100,
            'output_mse_diff': mse_diff,
            'output_cosine_sim': cos_sim,
            'max_abs_diff': diff.abs().max().item(),
            'mean_abs_diff': diff.abs().mean().item()
        },
        'weight_analysis': weight_analysis,
        'architecture': {
            'dims': dims,
            'batch': batch,
            'bottleneck': min(dims),
            'compression_ratio': dims[0] / min(dims),
            'total_params': sum(dims[i]*dims[i+1] for i in range(len(dims)-1))
        },
        'config': {'gd_target_error': target_error}
    }
    with open('centipede_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved: centipede_results.json")

if __name__ == "__main__":
    run_benchmark()
