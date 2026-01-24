#!/usr/bin/env python3
"""
llamarcp-wrapper - Simple CLI for running llama models
"""
import argparse
import sys
from pathlib import Path

from llamarcp import Llama


def main():
    parser = argparse.ArgumentParser(
        description="llamarcp-wrapper - Run inference with llama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple completion
  python -m llamarcp -m model.gguf -p "Hello, my name is"
  
  # Chat mode
  python -m llamarcp -m model.gguf --chat
  
  # With GPU acceleration
  python -m llamarcp -m model.gguf -ngl 99 -p "Once upon a time"
  
  # Adjust parameters
  python -m llamarcp -m model.gguf -p "Story:" -n 256 -t 0.7 --top-p 0.9
        """
    )
    
    # Model parameters
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model file")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=0, help="Number of layers to offload to GPU")
    parser.add_argument("-c", "--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("-b", "--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Generation parameters
    parser.add_argument("-p", "--prompt", default="", help="Prompt text")
    parser.add_argument("-n", "--n-predict", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-p sampling")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Random seed")
    
    # Mode
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream output")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}", file=sys.stderr)
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx_size,
        n_batch=args.batch_size,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.threads,
        verbose=args.verbose,
    )
    print("Model loaded!", file=sys.stderr)
    
    if args.chat:
        # Interactive chat mode
        print("\nChat mode - type 'exit' to quit\n", file=sys.stderr)
        messages = []
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ('exit', 'quit'):
                    break
                    
                messages.append({"role": "user", "content": user_input})
                
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=args.n_predict,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                    stream=args.stream,
                )
                
                assistant_message = ""
                print("Assistant: ", end="", flush=True)
                
                if args.stream:
                    for chunk in response:
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end="", flush=True)
                                assistant_message += content
                else:
                    assistant_message = response['choices'][0]['message']['content']
                    print(assistant_message)
                
                print()
                messages.append({"role": "assistant", "content": assistant_message})
                
            except KeyboardInterrupt:
                print("\nExiting...", file=sys.stderr)
                break
            except EOFError:
                break
    else:
        # Single completion mode
        if not args.prompt:
            print("Error: --prompt required for completion mode", file=sys.stderr)
            sys.exit(1)
        
        response = llm.create_completion(
            prompt=args.prompt,
            max_tokens=args.n_predict,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            stream=args.stream,
            seed=args.seed if args.seed >= 0 else None,
        )
        
        print(args.prompt, end="", flush=True)
        
        if args.stream:
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    print(text, end="", flush=True)
        else:
            print(response['choices'][0]['text'], end="", flush=True)
        
        print()


if __name__ == "__main__":
    main()
