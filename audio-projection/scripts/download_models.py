"""
Download required models for audio projection training.

This script downloads:
1. Meta-Llama-3.1-8B-Instruct (base LLM)
2. Qwen3-TTS-12Hz-1.7B-Base (TTS decoder)
"""

import os
import argparse
from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str, token: str = None):
    """
    Download model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save model
        token: Optional HuggingFace token for gated models
    """
    print(f"\nDownloading {repo_id}...")
    print(f"Saving to: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"✓ Successfully downloaded {repo_id}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for audio projection")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Base directory to save models",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (required for Llama)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip downloading LLM model",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip downloading TTS model",
    )
    
    args = parser.parse_args()
    
    # Model configurations
    models = {
        "llm": {
            "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "local_dir": os.path.join(args.output_dir, "llama-3.1-8b-instruct"),
            "skip": args.skip_llm,
            "requires_token": True,
        },
        "tts": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "local_dir": os.path.join(args.output_dir, "qwen3-tts-1.7b"),
            "skip": args.skip_tts,
            "requires_token": False,
        },
    }
    
    print("=" * 60)
    print("Audio Projection Model Downloader")
    print("=" * 60)
    
    # Check for HuggingFace token
    if not args.hf_token and not args.skip_llm:
        print("\n⚠ WARNING: Llama 3.1 requires HuggingFace authentication")
        print("Please provide --hf-token or set HF_TOKEN environment variable")
        print("You can skip LLM download with --skip-llm flag")
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("\n✗ No HuggingFace token found. Skipping LLM download.")
            models["llm"]["skip"] = True
        else:
            print("✓ Using HF_TOKEN from environment")
            args.hf_token = hf_token
    
    # Download models
    success_count = 0
    total_count = 0
    
    for model_type, config in models.items():
        if config["skip"]:
            print(f"\n⊗ Skipping {model_type.upper()} model")
            continue
        
        total_count += 1
        token = args.hf_token if config["requires_token"] else None
        
        if download_model(config["repo_id"], config["local_dir"], token):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{total_count} successful")
    print("=" * 60)
    
    if success_count == total_count:
        print("\n✓ All models downloaded successfully!")
        print(f"\nModels saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Prepare your training dataset")
        print("2. Configure training parameters in configs/default.yaml")
        print("3. Start training: python src/train.py --config configs/default.yaml")
    else:
        print("\n⚠ Some models failed to download")
        print("Please check errors above and retry")


if __name__ == "__main__":
    main()
