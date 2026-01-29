#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Extract embedding layer from any HuggingFace model
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Extract embedding layer from HuggingFace model')
    parser.add_argument('--model', required=True, help='Model path or HuggingFace model ID')
    parser.add_argument('--output', default='output/embedding.npy', help='Output file path')
    parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    parser.add_argument('--dtype', default='float32', help='Data type')
    parser.add_argument('--layer-name', default='model.embed_tokens', help='Embedding layer attribute path')
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    # Navigate to embedding layer using dot notation
    embedding = model
    for attr in args.layer_name.split('.'):
        embedding = getattr(embedding, attr)
    embedding = embedding.weight.data
    
    print(f"embedding shape: {embedding.shape}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'npy':
        np.save(output_path, embedding.cpu().numpy())
    else:
        torch.save(embedding, output_path)
    
    print(f"✓ Saved to {output_path}")

if __name__ == '__main__':
    main()
