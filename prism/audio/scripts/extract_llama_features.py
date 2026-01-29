#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Extract Llama hidden states from text.

Processes text through frozen Llama model and saves hidden states.
Optimized for Pascal GPUs (FP32, small batches).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import argparse


def extract_llama_features(
    dataset_path: str,
    output_dir: str,
    llm_path: str,
    device: str = "cuda",
    batch_size: int = 1,
):
    """
    Extract Llama hidden states from text.
    
    Args:
        dataset_path: Path to processed dataset
        output_dir: Output directory
        llm_path: Path to Llama model
        device: cuda or cpu
        batch_size: Batch size (keep 1-2 for Pascal)
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    
    # Create output directory
    llm_features_dir = output_path / "llm_hidden_states"
    llm_features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed index
    index_file = dataset_path / "processed_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Processed index not found: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        dataset_index = json.load(f)
    
    print("=" * 70)
    print("Extracting Llama Hidden States")
    print("=" * 70)
    print(f"  Dataset: {dataset_path}")
    print(f"  Samples: {len(dataset_index)}")
    print(f"  LLM: {llm_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    
    # Initialize CUDA first if needed
    if device == "cuda":
        torch.cuda.init()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load Llama
    print(f"\nLoading Llama model...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    
    # For Pascal: use FP32, load directly to device
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    if device == "cuda":
        model = model.cuda()
    
    model.eval()
    
    print(f"✓ Llama loaded")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    
    # Process samples
    updated_index = []
    
    for i in tqdm(range(0, len(dataset_index), batch_size), desc="Extracting features"):
        batch = dataset_index[i:i+batch_size]
        
        # Prepare texts
        texts = [item['text'] for item in batch]
        sample_ids = [item['id'] for item in batch]
        
        try:
            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)
            
            # Extract hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Save each sample
            for j, sample_id in enumerate(sample_ids):
                # Get this sample's hidden states (remove padding)
                attention_mask = inputs['attention_mask'][j]
                valid_len = attention_mask.sum().item()
                
                sample_hidden = hidden_states[j, :valid_len, :].cpu()
                
                # Validate
                if torch.isnan(sample_hidden).any():
                    print(f"\n⚠ NaN in hidden states for {sample_id}")
                    continue
                
                # Save
                feature_file = llm_features_dir / f"{sample_id}.pt"
                torch.save(sample_hidden, feature_file)
                
                # Update index
                item = batch[j].copy()
                item['llm_hidden_states_path'] = f"llm_hidden_states/{sample_id}.pt"
                item['num_tokens'] = valid_len
                item['llm_hidden_size'] = sample_hidden.shape[1]
                
                updated_index.append(item)
        
        except Exception as e:
            print(f"\n✗ Error processing batch {i}: {str(e)}")
            continue
    
    # Save final index
    final_index_file = output_path / "final_index.json"
    with open(final_index_file, 'w', encoding='utf-8') as f:
        json.dump(updated_index, f, indent=2, ensure_ascii=False)
    
    # Stats
    total_tokens = sum(item['num_tokens'] for item in updated_index)
    avg_tokens = total_tokens / len(updated_index)
    
    print("\n" + "=" * 70)
    print("Feature Extraction Complete")
    print("=" * 70)
    print(f"  Processed: {len(updated_index)} samples")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/sample: {avg_tokens:.1f}")
    print(f"  Hidden size: {updated_index[0]['llm_hidden_size']}")
    print(f"\n  Output: {output_path}")
    print(f"  Index: {final_index_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract Llama hidden states")
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to processed dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--llm-path',
        type=str,
        required=True,
        help='Path to Llama model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (keep 1-2 for Pascal GPUs)'
    )
    
    args = parser.parse_args()
    
    extract_llama_features(
        args.dataset_path,
        args.output_dir,
        args.llm_path,
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
