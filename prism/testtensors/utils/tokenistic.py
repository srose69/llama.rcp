#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Tokenistic - utility for tokenizer mapping and comparison.

Functions:
- Compare vocabularies of two tokenizers
- Find common tokens
- Exclude garbage/special tokens
- Count differences and losses in mapping
"""

import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from typing import Tuple, Dict, Set, List
import json
from pathlib import Path


def load_tokenizers(upper_model_path: str, lower_model_path: str) -> Tuple:
    """
    Load tokenizers for both models.
    
    Returns:
        upper_tokenizer, lower_tokenizer
    """
    upper_tokenizer = AutoTokenizer.from_pretrained(
        upper_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    lower_tokenizer = AutoTokenizer.from_pretrained(
        lower_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    return upper_tokenizer, lower_tokenizer


def get_special_tokens(tokenizer) -> Set[str]:
    """
    Get all special tokens from tokenizer.
    
    Returns:
        Set of special tokens
    """
    special = set()
    
    # Known special tokens
    if tokenizer.bos_token:
        special.add(tokenizer.bos_token)
    if tokenizer.eos_token:
        special.add(tokenizer.eos_token)
    if tokenizer.pad_token:
        special.add(tokenizer.pad_token)
    if tokenizer.unk_token:
        special.add(tokenizer.unk_token)
    if tokenizer.sep_token:
        special.add(tokenizer.sep_token)
    if tokenizer.cls_token:
        special.add(tokenizer.cls_token)
    if tokenizer.mask_token:
        special.add(tokenizer.mask_token)
    
    # Additional special tokens
    if hasattr(tokenizer, 'additional_special_tokens'):
        special.update(tokenizer.additional_special_tokens)
    
    return special


def filter_garbage_tokens(vocab: Dict[str, int], special_tokens: Set[str]) -> Dict[str, int]:
    """
    Filter garbage tokens from vocabulary.
    
    Garbage tokens:
    - Special tokens (BOS, EOS, PAD, UNK, etc.)
    - Tokens with control characters
    - Empty tokens
    
    Args:
        vocab: Dictionary {token: token_id}
        special_tokens: Set of special tokens
        
    Returns:
        Filtered dictionary
    """
    filtered = {}
    
    for token, token_id in vocab.items():
        # Skip special tokens
        if token in special_tokens:
            continue
            
        # Skip empty
        if not token or len(token) == 0:
            continue
            
        # Skip tokens with only control characters
        if all(ord(c) < 32 or ord(c) == 127 for c in token):
            continue
            
        filtered[token] = token_id
    
    return filtered


def find_common_tokens(upper_vocab: Dict[str, int], 
                       lower_vocab: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """
    Find common tokens between two vocabularies.
    
    Args:
        upper_vocab: Upper model dictionary {token: token_id}
        lower_vocab: Lower model dictionary {token: token_id}
        
    Returns:
        Dict {token: (upper_id, lower_id)} for common tokens
    """
    common = {}
    
    upper_tokens = set(upper_vocab.keys())
    lower_tokens = set(lower_vocab.keys())
    
    common_token_set = upper_tokens & lower_tokens
    
    for token in common_token_set:
        common[token] = (upper_vocab[token], lower_vocab[token])
    
    return common


def analyze_tokenizers(upper_model_path: str, lower_model_path: str) -> Dict:
    """
    Complete analysis of differences between tokenizers.
    
    Returns:
        Dict with statistics and mapping
    """
    print(f"\n🔍 Analyzing tokenizers...")
    print(f"  Upper: {upper_model_path}")
    print(f"  Lower: {lower_model_path}")
    
    # Load tokenizers
    upper_tok, lower_tok = load_tokenizers(upper_model_path, lower_model_path)
    
    # Get vocabularies
    upper_vocab = upper_tok.get_vocab()
    lower_vocab = lower_tok.get_vocab()
    
    print(f"\n📊 Vocabulary sizes:")
    print(f"  Upper: {len(upper_vocab):,} tokens")
    print(f"  Lower: {len(lower_vocab):,} tokens")
    
    # Get special tokens
    upper_special = get_special_tokens(upper_tok)
    lower_special = get_special_tokens(lower_tok)
    
    print(f"\n🔖 Special tokens:")
    print(f"  Upper: {len(upper_special)} special tokens")
    print(f"  Lower: {len(lower_special)} special tokens")
    
    # Filter garbage
    upper_clean = filter_garbage_tokens(upper_vocab, upper_special)
    lower_clean = filter_garbage_tokens(lower_vocab, lower_special)
    
    print(f"\n🧹 After filtering garbage:")
    print(f"  Upper: {len(upper_clean):,} clean tokens")
    print(f"  Lower: {len(lower_clean):,} clean tokens")
    
    # Find common tokens
    common = find_common_tokens(upper_clean, lower_clean)
    
    print(f"\n✅ Common tokens: {len(common):,}")
    print(f"  Coverage: {len(common)/len(upper_clean)*100:.2f}% of upper vocab")
    print(f"  Coverage: {len(common)/len(lower_clean)*100:.2f}% of lower vocab")
    
    # Unique tokens
    upper_unique = set(upper_clean.keys()) - set(common.keys())
    lower_unique = set(lower_clean.keys()) - set(common.keys())
    
    print(f"\n❌ Unique tokens (lost in mapping):")
    print(f"  Upper only: {len(upper_unique):,}")
    print(f"  Lower only: {len(lower_unique):,}")
    
    # Result
    result = {
        'upper_vocab_size': len(upper_vocab),
        'lower_vocab_size': len(lower_vocab),
        'upper_clean_size': len(upper_clean),
        'lower_clean_size': len(lower_clean),
        'common_tokens': len(common),
        'upper_unique': len(upper_unique),
        'lower_unique': len(lower_unique),
        'upper_coverage': len(common) / len(upper_clean),
        'lower_coverage': len(common) / len(lower_clean),
        'common_mapping': common,
        'upper_special_tokens': list(upper_special),
        'lower_special_tokens': list(lower_special),
    }
    
    return result


def create_token_mapping(upper_model_path: str, 
                         lower_model_path: str,
                         output_file: str = None) -> Tuple[List[int], List[int]]:
    """
    Create token mapping for worker training.
    
    Returns:
        upper_indices: List[int] - token indices in upper model
        lower_indices: List[int] - corresponding indices in lower model
    """
    result = analyze_tokenizers(upper_model_path, lower_model_path)
    
    common_mapping = result['common_mapping']
    
    # Create index lists
    upper_indices = []
    lower_indices = []
    
    for token, (upper_id, lower_id) in sorted(common_mapping.items()):
        upper_indices.append(upper_id)
        lower_indices.append(lower_id)
    
    print(f"\n💾 Token mapping created:")
    print(f"  {len(upper_indices):,} token pairs")
    
    # Save if needed
    if output_file:
        mapping_data = {
            'num_tokens': len(upper_indices),
            'upper_indices': upper_indices,
            'lower_indices': lower_indices,
            'statistics': {
                k: v for k, v in result.items() 
                if k != 'common_mapping'
            }
        }
        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"  Saved to: {output_file}")
    
    return upper_indices, lower_indices


def tokenalyze(model_path: str, output_file: str = None) -> Dict[str, float]:
    """
    Analyze model and extract weights for each token from embed_tokens.
    Uses direct loading from safetensors with FP64 precision.
    
    Args:
        model_path: Path to model
        output_file: File to save token→weight mapping (JSON)
        
    Returns:
        Dict {token_string: weight_value}
    """
    print(f"\n🔬 Tokenalyze: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Load embed_tokens directly from safetensors
    print(f"  Loading embed_tokens from safetensors...")
    model_dir = Path(model_path)
    safetensors_files = list(model_dir.glob("model-*.safetensors"))
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    embed_tokens = None
    for st_file in safetensors_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if 'embed_tokens.weight' in key:
                    print(f"    Found key: {key} in {st_file.name}")
                    tensor = f.get_tensor(key)
                    # FP64 for precision
                    embed_tokens = tensor.to(torch.float64)
                    break
            if embed_tokens is not None:
                break
    
    if embed_tokens is None:
        raise ValueError(f"Cannot find embed_tokens.weight in safetensors files")
    
    print(f"  Embed tokens shape: {embed_tokens.shape}")
    print(f"  Precision: {embed_tokens.dtype}")
    
    # Get vocabulary
    vocab = tokenizer.get_vocab()
    
    # Create token → weight mapping (using vector norm)
    token_weights = {}
    
    for token_str, token_id in vocab.items():
        if token_id < embed_tokens.shape[0]:
            # Use vector norm as token "weight"
            weight = float(torch.norm(embed_tokens[token_id]).item())
            token_weights[token_str] = weight
    
    print(f"\n✅ Analyzed {len(token_weights):,} tokens")
    print(f"  Weight range: [{min(token_weights.values()):.6f}, {max(token_weights.values()):.6f}]")
    
    # Save if needed
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(token_weights, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved to: {output_file}")
    
    return token_weights


def tokenmap(model_a_path: str, model_b_path: str, output_file: str = None) -> Dict:
    """
    Map tokens between models A and B by strings.
    
    Args:
        model_a_path: Path to model A
        model_b_path: Path to model B
        output_file: File to save mapping (JSON)
        
    Returns:
        Dict with tokenA → tokenB mapping
    """
    print(f"\n🗺️  Tokenmap:")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    
    # Загрузить токенизаторы
    tokenizer_a = AutoTokenizer.from_pretrained(
        model_a_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        model_b_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()
    
    print(f"\n📊 Vocabulary sizes:")
    print(f"  Model A: {len(vocab_a):,} tokens")
    print(f"  Model B: {len(vocab_b):,} tokens")
    
    # Find common tokens (string intersection)
    tokens_a = set(vocab_a.keys())
    tokens_b = set(vocab_b.keys())
    
    common_tokens = tokens_a & tokens_b
    
    # Create mapping
    token_mapping = {}
    for token in common_tokens:
        token_mapping[token] = {
            'token_a_id': vocab_a[token],
            'token_b_id': vocab_b[token],
            'token_string': token
        }
    
    print(f"\n✅ Mapped {len(token_mapping):,} common tokens")
    print(f"  Coverage A: {len(common_tokens)/len(tokens_a)*100:.2f}%")
    print(f"  Coverage B: {len(common_tokens)/len(tokens_b)*100:.2f}%")
    
    # Result
    result = {
        'model_a_vocab_size': len(vocab_a),
        'model_b_vocab_size': len(vocab_b),
        'common_tokens_count': len(common_tokens),
        'mapping': token_mapping
    }
    
    # Save if needed
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved to: {output_file}")
    
    return result


def difftokmap(model_a_path: str, model_b_path: str, output_file: str = None) -> Dict:
    """
    Create file with token differences between model A and model B.
    
    Shows:
    - Which tokens exist in both models (by presence, not weights)
    - Which tokens only in A
    - Which tokens only in B
    
    Args:
        model_a_path: Path to model A
        model_b_path: Path to model B
        output_file: File to save differences (JSON)
        
    Returns:
        Dict with lists common_tokens, only_a, only_b
    """
    print(f"\n🔀 Difftokmap:")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    
    # Загрузить токенизаторы
    tokenizer_a = AutoTokenizer.from_pretrained(
        model_a_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        model_b_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()
    
    # Get token sets
    tokens_a = set(vocab_a.keys())
    tokens_b = set(vocab_b.keys())
    
    # Calculate intersections and unique
    common_tokens = sorted(list(tokens_a & tokens_b))
    only_a = sorted(list(tokens_a - tokens_b))
    only_b = sorted(list(tokens_b - tokens_a))
    
    print(f"\n Token differences:")
    print(f"  Common tokens (in both): {len(common_tokens):,}")
    print(f"  Only in A: {len(only_a):,}")
    print(f"  Only in B: {len(only_b):,}")
    print(f"  Overlap: {len(common_tokens)/(len(tokens_a))*100:.2f}% of A")
    print(f"  Overlap: {len(common_tokens)/(len(tokens_b))*100:.2f}% of B")
    
    # Result
    result = {
        'model_a': model_a_path,
        'model_b': model_b_path,
        'model_a_vocab_size': len(vocab_a),
        'model_b_vocab_size': len(vocab_b),
        'common_tokens_count': len(common_tokens),
        'only_a_count': len(only_a),
        'only_b_count': len(only_b),
        'common_tokens': common_tokens,
        'only_a': only_a,
        'only_b': only_b
    }
    
    # Save if needed
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved to: {output_file}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tokenistic - tokenizer analysis and mapping')
    parser.add_argument('--mode', required=True, 
                        choices=['analyze', 'mapping', 'tokenalyze', 'tokenmap', 'difftokmap'],
                        help='Operation mode')
    parser.add_argument('--model', help='Path to model (for tokenalyze)')
    parser.add_argument('--model-a', help='Path to model A (for tokenmap/difftokmap)')
    parser.add_argument('--model-b', help='Path to model B (for tokenmap/difftokmap)')
    parser.add_argument('--upper-model', help='Path to upper model (for analyze/mapping)')
    parser.add_argument('--lower-model', help='Path to lower model (for analyze/mapping)')
    parser.add_argument('--output', help='File to save result (JSON)')
    
    args = parser.parse_args()
    
    if args.mode == 'tokenalyze':
        if not args.model:
            parser.error("--model required for tokenalyze mode")
        tokenalyze(args.model, args.output)
        
    elif args.mode == 'tokenmap':
        if not args.model_a or not args.model_b:
            parser.error("--model-a and --model-b required for tokenmap mode")
        tokenmap(args.model_a, args.model_b, args.output)
        
    elif args.mode == 'difftokmap':
        if not args.model_a or not args.model_b:
            parser.error("--model-a and --model-b required for difftokmap mode")
        difftokmap(args.model_a, args.model_b, args.output)
        
    elif args.mode == 'analyze':
        if not args.upper_model or not args.lower_model:
            parser.error("--upper-model and --lower-model required for analyze mode")
        analyze_tokenizers(args.upper_model, args.lower_model)
        
    elif args.mode == 'mapping':
        if not args.upper_model or not args.lower_model:
            parser.error("--upper-model and --lower-model required for mapping mode")
        create_token_mapping(args.upper_model, args.lower_model, args.output)
