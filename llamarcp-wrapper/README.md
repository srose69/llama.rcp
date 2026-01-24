# llama.rcp Python Wrapper

Python ctypes bindings and high-level API for llama.rcp (llama.cpp fork).

## Features

- **Simple API**: High-level `Llama` class for easy model loading and inference
- **OpenAI-compatible**: Compatible with OpenAI API format for completions and chat
- **GPU Acceleration**: Full CUDA support (sm_61/Pascal tested)
- **Streaming**: Support for streaming completions
- **Embeddings**: Generate text embeddings
- **Chat Formats**: Multiple chat templates (llama-2, chatml, etc.)
- **CLI**: Command-line interface for quick testing

## Installation

### Prerequisites

The wrapper requires the compiled llama.rcp libraries:

```bash
# Build llama.rcp first (from repository root)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build --config Release -j$(nproc)
```

### Setup

The wrapper is ready to use - libraries are symlinked in `wrapper/lib/`.

Alternatively, set the library path manually:

```bash
export LLAMA_CPP_LIB_PATH=/path/to/llama.rcp/build/bin
```

## Usage

### Command Line Interface

#### Simple Completion

```bash
# Basic text generation
python -m llamarcp -m model.gguf -p "Once upon a time"

# With GPU acceleration (all layers)
python -m llamarcp -m model.gguf -ngl 99 -p "Hello, world"

# Adjust parameters
python -m llamarcp -m model.gguf -p "Story:" \
  -n 256 \          # Generate 256 tokens
  -t 0.7 \          # Temperature 0.7
  --top-p 0.9 \     # Top-p sampling
  --repeat-penalty 1.1
```

#### Interactive Chat

```bash
# Start chat mode
python -m llamarcp -m model.gguf --chat

# With GPU
python -m llamarcp -m model.gguf --chat -ngl 99
```

#### Full Example

```bash
python -m llamarcp \
  -m /path/to/model.gguf \
  -ngl 99 \
  -c 4096 \
  -p "Write a short story about a robot:" \
  -n 512 \
  -t 0.8 \
  --top-k 40 \
  --top-p 0.95
```

### Python API

#### Basic Completion

```python
from llamarcp import Llama

# Load model
llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    n_gpu_layers=99,  # Use GPU
)

# Generate text
output = llm(
    "Once upon a time",
    max_tokens=128,
    temperature=0.8,
    top_p=0.95,
)
print(output['choices'][0]['text'])
```

#### Streaming Completion

```python
from llamarcp import Llama

llm = Llama(model_path="model.gguf", n_gpu_layers=99)

stream = llm(
    "Write a poem about AI:",
    max_tokens=256,
    stream=True,
)

for chunk in stream:
    print(chunk['choices'][0]['text'], end='', flush=True)
```

#### Chat Completion

```python
from llamarcp import Llama

llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=99,
    chat_format="llama-2",  # or "chatml", etc.
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response['choices'][0]['message']['content'])
```

#### Embeddings

```python
from llamarcp import Llama

llm = Llama(
    model_path="model.gguf",
    embedding=True,
    n_gpu_layers=99,
)

embeddings = llm.create_embedding("Hello, world!")
print(embeddings['data'][0]['embedding'][:5])  # First 5 dimensions
```

## API Reference

### Llama Class

#### Constructor Parameters

**Model Parameters:**
- `model_path` (str): Path to GGUF model file
- `n_gpu_layers` (int): Number of layers to offload to GPU (default: 0, -1 for all)
- `n_ctx` (int): Context size (default: 512)
- `n_batch` (int): Batch size for prompt processing (default: 512)
- `n_threads` (int): Number of threads (default: auto)
- `use_mmap` (bool): Use memory mapping (default: True)
- `use_mlock` (bool): Lock model in RAM (default: False)

**Sampling Parameters:**
- `seed` (int): Random seed (default: -1 for random)
- `verbose` (bool): Verbose output (default: True)

**Advanced:**
- `embedding` (bool): Embedding mode (default: False)
- `logits_all` (bool): Return logits for all tokens (default: False)
- `chat_format` (str): Chat format template
- `lora_path` (str): Path to LoRA adapter
- `flash_attn` (bool): Enable flash attention (default: False)

#### Methods

**`__call__(prompt, **kwargs)`**
Simple completion with prompt string.

**`create_completion(prompt, **kwargs)`**
Generate completion with full control.

Parameters:
- `prompt` (str | List[int]): Prompt text or token IDs
- `max_tokens` (int): Maximum tokens to generate (default: 16)
- `temperature` (float): Sampling temperature (default: 0.8)
- `top_k` (int): Top-k sampling (default: 40)
- `top_p` (float): Top-p/nucleus sampling (default: 0.95)
- `min_p` (float): Min-p sampling (default: 0.05)
- `repeat_penalty` (float): Repeat penalty (default: 1.1)
- `frequency_penalty` (float): Frequency penalty (default: 0.0)
- `presence_penalty` (float): Presence penalty (default: 0.0)
- `stop` (str | List[str]): Stop sequences
- `stream` (bool): Stream output (default: False)
- `seed` (int): Random seed
- `grammar` (LlamaGrammar): Grammar constraint

**`create_chat_completion(messages, **kwargs)`**
Chat completion with message history.

Parameters:
- `messages` (List[dict]): Chat messages with `role` and `content`
- Same generation parameters as `create_completion`

**`create_embedding(input)`**
Generate embeddings (requires `embedding=True`).

**`tokenize(text, add_bos=True, special=False)`**
Tokenize text to token IDs.

**`detokenize(tokens, special=False)`**
Convert tokens back to text.

## Environment Variables

- `LLAMA_CPP_LIB_PATH`: Override library search path
- `CUDA_VISIBLE_DEVICES`: Select GPU devices

## Tested Models

The wrapper has been tested with:
- ✅ LFM2.5-1.2B-Instruct-BF16.gguf
- ✅ Llama 2/3 models
- ✅ Qwen models
- ✅ Mistral models

Any GGUF model should work.

## Performance Tips

1. **GPU Acceleration**: Use `-ngl 99` or `n_gpu_layers=-1` to offload all layers
2. **Context Size**: Increase `n_ctx` for longer conversations
3. **Batch Size**: Increase `n_batch` for faster prompt processing
4. **Flash Attention**: Enable with `flash_attn=True` for faster inference
5. **Threads**: Set `n_threads` to match your CPU cores

## Troubleshooting

### Library not found

```
FileNotFoundError: Shared library with base name 'llama' not found
```

**Solution**: Ensure libraries are built and symlinked:
```bash
cd /path/to/llama.rcp
ls -la wrapper/lib/  # Should show libllama.so*, libggml*.so*
```

Or set the path manually:
```bash
export LLAMA_CPP_LIB_PATH=/path/to/llama.rcp/build/bin
```

### CUDA errors

```
CUDA error: invalid device ordinal
```

**Solution**: Check GPU availability and select correct device:
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

### Out of memory

**Solution**: Reduce `n_ctx`, offload fewer layers, or use quantized model.

## Architecture

```
wrapper/
├── __init__.py           # Package entry point
├── __main__.py           # CLI interface
├── llama.py              # High-level Llama class
├── llamarcp.py          # C API ctypes bindings
├── _internals.py         # Internal wrapper classes
├── _ctypes_extensions.py # Library loader
├── llama_chat_format.py  # Chat templates
├── llama_grammar.py      # Grammar constraints
├── llama_types.py        # OpenAI-compatible types
└── lib/                  # Compiled libraries (symlinks)
    ├── libllama.so
    ├── libggml-base.so
    ├── libggml-cpu.so
    └── libggml-cuda.so
```

## License

Same as llama.cpp (MIT License).

## Credits

- Based on llamarcp by Andrei Betlen
- llama.cpp by Georgi Gerganov
- llama.rcp modifications
