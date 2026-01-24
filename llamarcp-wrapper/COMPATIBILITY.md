# Compatibility Issues - llama.rcp Python Wrapper

## Current Status: ⚠️ INCOMPATIBLE

The Python wrapper code is **NOT compatible** with the current llama.rcp C++ library.

## Problem

The wrapper code uses deprecated API functions that have been **removed** from llama.cpp:

### Missing Functions

1. **`llama_get_kv_self()`** (line 1408 in llamarcp.py)
   - Used in: `_internals.py` for KV cache access
   - Status: DEPRECATED and removed from C library
   - Error: `undefined symbol: llama_get_kv_self`

2. **`llama_kv_self_clear()`** (line 2071 in llamarcp.py)
   - Used in: `llama.py` for cache clearing
   - Status: DEPRECATED and removed from C library
   - Error: `undefined symbol: llama_kv_self_clear`

### Version Mismatch

- **Wrapper Code**: Based on llama.cpp with deprecated API still present
- **C++ Library**: Current llama.cpp (commit 0bc2f74) with deprecated API removed
- **Gap**: Wrapper expects old API, library only has new API

## Workarounds

### Option 1: Use llama-cli Directly (Recommended) ✅

The C++ binaries work perfectly:

```bash
# Direct inference
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -p "Hello, world" \
  -n 128

# Server mode
./build/bin/llama-server \
  -m model.gguf \
  -ngl 99 \
  --port 8080
```

**Pros:**
- Works immediately
- Full CUDA support
- No compatibility issues
- Native performance

**Cons:**
- No Python API
- CLI only

### Option 2: Use Official llama-cpp-python

Install the official Python bindings:

```bash
# Install from PyPI
# TODO: Not published yet
# pip install llama-cpp-python

# With CUDA support
# CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61" \
#   pip install llama-cpp-python --force-reinstall --no-cache-dir

# Use llama.rcp libraries
# export LLAMA_CPP_LIB_PATH=/path/to/llama.rcp/build/bin
# python3 -c "from llama_cpp import Llama; ..."
```

**Pros:**
- Maintained and up-to-date
- Compatible with current llama.cpp
- Full Python API

**Cons:**
- External dependency
- May need rebuild for custom backends
- **Not published to PyPI yet**

### Option 3: Update Wrapper Code (Advanced)

The wrapper needs extensive updates to use the new API:

**Required Changes:**
1. Replace `llama_get_kv_self()` with new memory API
2. Replace `llama_kv_self_clear()` with `llama_memory_clear()`
3. Update all KV cache access patterns
4. Fix 50+ deprecated function calls
5. Update type definitions for new structs

**Estimated Effort:** Several hours of development + testing

**Files to Update:**
- `llamarcp.py` (~4375 lines)
- `_internals.py` (~857 lines)
- `llama.py` (~2423 lines)

## Recommended Path Forward

**For immediate use:**
1. Use `./build/bin/llama-cli` for testing
2. Use `./build/bin/llama-server` for API access
3. ~~Install official `llama-cpp-python` if Python API needed~~ (not on PyPI yet)

**For long-term:**
1. Track llama.cpp updates
2. Sync wrapper code with upstream llamarcp
3. Or contribute fixes to make wrapper compatible

## Current Working Setup

```bash
# What WORKS:
./build/bin/llama-cli -m model.gguf -ngl 99 -p "Hello" -n 20
./build/bin/llama-server -m model.gguf -ngl 99 --port 8080
./build/bin/llama-bench ...

# What DOESN'T work:
python3 -m llamarcp -m model.gguf ...  # Symbol errors
from llamarcp import Llama             # Symbol errors
```

## Testing Done

✅ C++ binaries compiled successfully
✅ CUDA backend working (sm_61)
✅ llama-cli inference tested
✅ All 50+ tools compiled
❌ Python wrapper incompatible with current library

## Next Steps

User should decide:
1. Use CLI tools (ready now)
2. Install official llamarcp (pip install)
3. Update wrapper code (requires development)

The C++ build is **complete and working**. The Python wrapper requires either using official bindings or significant code updates.
