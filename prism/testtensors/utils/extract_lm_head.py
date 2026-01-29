#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Extract lm_head from any HuggingFace model
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Extract lm_head from HuggingFace model')
    parser.add_argument('--model', required=True, help='Model path or HuggingFace model ID')
    parser.add_argument('--output', default='output/lm_head.npy', help='Output file path')
    parser.add_argument('--format', choices=['npy', 'pt'], default='npy', help='Output format')
    parser.add_argument('--dtype', default='float32', help='Data type')
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    lm_head = model.lm_head.weight.data
    print(f"lm_head shape: {lm_head.shape}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'npy':
        np.save(output_path, lm_head.cpu().numpy())
    else:
        torch.save(lm_head, output_path)
    
    print(f"✓ Saved to {output_path}")

if __name__ == '__main__':
    main()
