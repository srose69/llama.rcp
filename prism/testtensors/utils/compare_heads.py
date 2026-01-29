#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Compare two weight matrices (from files or models)
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import argparse
from pathlib import Path

def load_matrix(source, layer_path=None):
    """Load matrix from file or model"""
    source_path = Path(source)
    
    if source_path.exists():
        # Load from file
        if source_path.suffix == '.npy':
            return np.load(source_path)
        elif source_path.suffix == '.pt':
            return torch.load(source_path).cpu().numpy()
        else:
            raise ValueError(f"Unsupported file format: {source_path.suffix}")
    else:
        # Try loading as HuggingFace model
        print(f"   Loading model: {source}")
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float32,
            device_map='cpu',
            low_cpu_mem_usage=True
        )
        
        if layer_path:
            layer = model
            for attr in layer_path.split('.'):
                layer = getattr(layer, attr)
            return layer.weight.data.cpu().numpy()
        else:
            # Default to lm_head
            return model.lm_head.weight.data.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Compare two weight matrices')
    parser.add_argument('--matrix1', required=True, help='First matrix (file path or model ID)')
    parser.add_argument('--matrix2', required=True, help='Second matrix (file path or model ID)')
    parser.add_argument('--layer1', help='Layer path for matrix1 (if model)')
    parser.add_argument('--layer2', help='Layer path for matrix2 (if model)')
    parser.add_argument('--name1', default='Matrix 1', help='Name for first matrix')
    parser.add_argument('--name2', default='Matrix 2', help='Name for second matrix')
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARING WEIGHT MATRICES")
    print("="*80)
    
    print(f"\n1. Loading {args.name1}...")
    matrix1 = load_matrix(args.matrix1, args.layer1)
    print(f"   Shape: {matrix1.shape}")
    print(f"   Dtype: {matrix1.dtype}")
    print(f"   Size: {matrix1.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n2. Loading {args.name2}...")
    matrix2 = load_matrix(args.matrix2, args.layer2)
    print(f"   Shape: {matrix2.shape}")
    print(f"   Dtype: {matrix2.dtype}")
    print(f"   Size: {matrix2.nbytes / 1024 / 1024:.2f} MB")

    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nShape comparison:")
    print(f"  {args.name1}: {matrix1.shape}")
    print(f"  {args.name2}: {matrix2.shape}")
    print(f"  Shapes match: {matrix1.shape == matrix2.shape}")
    
    print(f"\nDtype comparison:")
    print(f"  {args.name1}: {matrix1.dtype}")
    print(f"  {args.name2}: {matrix2.dtype}")
    
    print(f"\nStatistics comparison:")
    print(f"\n  {args.name1}:")
    print(f"    Min:    {matrix1.min():.6f}")
    print(f"    Max:    {matrix1.max():.6f}")
    print(f"    Mean:   {matrix1.mean():.6f}")
    print(f"    Std:    {matrix1.std():.6f}")
    
    print(f"\n  {args.name2}:")
    print(f"    Min:    {matrix2.min():.6f}")
    print(f"    Max:    {matrix2.max():.6f}")
    print(f"    Mean:   {matrix2.mean():.6f}")
    print(f"    Std:    {matrix2.std():.6f}")
    
    if matrix1.shape == matrix2.shape:
        print(f"\nChecking if identical...")
        are_equal = np.array_equal(matrix1, matrix2)
        print(f"  Arrays identical: {are_equal}")
        
        if not are_equal:
            diff = np.abs(matrix1 - matrix2)
            print(f"\n  Difference statistics:")
            print(f"    Max diff:  {diff.max():.10f}")
            print(f"    Mean diff: {diff.mean():.10f}")
            print(f"    Median diff: {np.median(diff):.10f}")
            
            close = np.allclose(matrix1, matrix2, rtol=1e-5, atol=1e-8)
            print(f"  Close (rtol=1e-5, atol=1e-8): {close}")
    else:
        print(f"\n⚠ Shapes don't match - cannot compare element-wise")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if matrix1.shape != matrix2.shape:
        print("❌ Different shapes - these are from different model architectures")
    elif np.array_equal(matrix1, matrix2):
        print("✅ IDENTICAL! Matrices are exactly the same")
    else:
        print("❌ Same shape but different values")
    
    print("="*80)

if __name__ == '__main__':
    main()
