# Python Wrapper Migration TODO

**Generated:** 2026-01-24  
**Purpose:** Track wrapper function compatibility with current llama.cpp API  
**Reference:** DEFS_ROADMAP.md for complete API specification

---

## Status Legend

- **✅ DONE** - Function works correctly with current API
- **⚠️ NEED** - Function exists but needs update/fix
- **❌ MISSING** - Function not implemented in wrapper
- **🗑️ DEPRECATED** - Function should be removed (deprecated in llama.cpp)
- **⚡ CRITICAL** - Blocking issue preventing wrapper from working

---

## Critical Issues (BLOCKING)

### ⚡ CRITICAL: Removed Functions

These functions are called by wrapper but **DO NOT EXIST** in current llama.cpp:

| Function | Status | Location | Issue | Action Required |
|----------|--------|----------|-------|-----------------|
| `llama_get_kv_self` | ✅ **DONE** | `llamarcp_wrapper.py:1443` | Removed binding | Replaced with comment, use `llama_get_memory()` |
| `llama_kv_self_clear` | ⚡ **REMOVED** | Used in cache management | Symbol not found | Replace with `llama_memory_clear()` |
| `llama_kv_cache_seq_rm` | ⚡ **REMOVED** | Cache operations | Symbol not found | Replace with `llama_memory_seq_rm()` |
| `llama_kv_cache_seq_cp` | ⚡ **REMOVED** | Cache operations | Symbol not found | Replace with `llama_memory_seq_cp()` |
| `llama_kv_cache_seq_keep` | ⚡ **REMOVED** | Cache operations | Symbol not found | Replace with `llama_memory_seq_keep()` |
| `llama_kv_cache_seq_add` | ⚡ **REMOVED** | Cache operations | Symbol not found | Replace with `llama_memory_seq_add()` |
| `llama_kv_cache_seq_div` | ⚡ **REMOVED** | Cache operations | Symbol not found | Replace with `llama_memory_seq_div()` |

**Impact:** Wrapper cannot import due to missing symbols  
**Priority:** P0 - Must fix before wrapper can work

---

## Backend Initialization

| Function | Status | Notes |
|----------|--------|-------|
| `llama_backend_init` | ✅ DONE | Working |
| `llama_backend_free` | ✅ DONE | Working |
| `llama_numa_init` | ✅ DONE | Working |

---

## Model Management

| Function | Status | Notes |
|----------|--------|-------|
| `llama_model_default_params` | ✅ DONE | Working |
| `llama_model_load_from_file` | ✅ DONE | Working |
| `llama_model_load_from_splits` | ✅ DONE | Working |
| `llama_model_save_to_file` | ✅ DONE | Working |
| `llama_model_free` | ✅ DONE | Working |
| `llama_load_model_from_file` | 🗑️ DEPRECATED | Remove - use `llama_model_load_from_file` |
| `llama_free_model` | 🗑️ DEPRECATED | Remove - use `llama_model_free` |
| `llama_model_quantize` | ⚠️ NEED | Exists but needs verification |

---

## Model Metadata

| Function | Status | Notes |
|----------|--------|-------|
| `llama_model_get_vocab` | ✅ DONE | Working |
| `llama_model_rope_type` | ✅ DONE | Working |
| `llama_model_n_ctx_train` | ✅ DONE | Working |
| `llama_model_n_embd` | ✅ DONE | Working |
| `llama_model_n_embd_inp` | ❌ MISSING | Not in wrapper |
| `llama_model_n_embd_out` | ❌ MISSING | Not in wrapper |
| `llama_model_n_layer` | ✅ DONE | Working |
| `llama_model_n_head` | ✅ DONE | Working |
| `llama_model_n_head_kv` | ❌ MISSING | Not in wrapper |
| `llama_model_n_swa` | ❌ MISSING | Not in wrapper |
| `llama_model_rope_freq_scale_train` | ❌ MISSING | Not in wrapper |
| `llama_model_n_cls_out` | ❌ MISSING | Not in wrapper |
| `llama_model_cls_label` | ❌ MISSING | Not in wrapper |
| `llama_model_meta_val_str` | ⚠️ NEED | Exists but needs verification |
| `llama_model_meta_count` | ⚠️ NEED | Exists but needs verification |
| `llama_model_meta_key_str` | ❌ MISSING | Not in wrapper |
| `llama_model_meta_key_by_index` | ⚠️ NEED | Exists but needs verification |
| `llama_model_meta_val_str_by_index` | ⚠️ NEED | Exists but needs verification |
| `llama_model_desc` | ⚠️ NEED | Exists but needs verification |
| `llama_model_size` | ⚠️ NEED | Exists but needs verification |
| `llama_model_chat_template` | ❌ MISSING | Not in wrapper |
| `llama_model_n_params` | ⚠️ NEED | Exists but needs verification |
| `llama_model_has_encoder` | ❌ MISSING | Not in wrapper |
| `llama_model_has_decoder` | ❌ MISSING | Not in wrapper |
| `llama_model_decoder_start_token` | ❌ MISSING | Not in wrapper |
| `llama_model_is_recurrent` | ❌ MISSING | Not in wrapper |
| `llama_model_is_hybrid` | ❌ MISSING | Not in wrapper |
| `llama_model_is_diffusion` | ❌ MISSING | Not in wrapper |

---

## Context Management

| Function | Status | Notes |
|----------|--------|-------|
| `llama_context_default_params` | ✅ DONE | Working |
| `llama_init_from_model` | ✅ DONE | Working |
| `llama_free` | ✅ DONE | Working |
| `llama_new_context_with_model` | 🗑️ DEPRECATED | Remove - use `llama_init_from_model` |
| `llama_params_fit` | ❌ MISSING | Not in wrapper |
| `llama_attach_threadpool` | ❌ MISSING | Not in wrapper |
| `llama_detach_threadpool` | ❌ MISSING | Not in wrapper |

---

## Context Queries

| Function | Status | Notes |
|----------|--------|-------|
| `llama_n_ctx` | ✅ DONE | Working |
| `llama_n_ctx_seq` | ❌ MISSING | Not in wrapper |
| `llama_n_batch` | ✅ DONE | Working |
| `llama_n_ubatch` | ✅ DONE | Working |
| `llama_n_seq_max` | ✅ DONE | Working |
| `llama_get_model` | ✅ DONE | Working |
| `llama_get_memory` | ✅ DONE | **NEW API** - Working |
| `llama_pooling_type` | ✅ DONE | Working |

---

## Memory Management (NEW API - CRITICAL)

**Status:** ⚡ **CRITICAL MIGRATION NEEDED**

Old KV cache API has been completely removed. Must migrate to new Memory API.

| Function | Status | Notes |
|----------|--------|-------|
| `llama_memory_clear` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_self_clear` |
| `llama_memory_seq_rm` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_cache_seq_rm` |
| `llama_memory_seq_cp` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_cache_seq_cp` |
| `llama_memory_seq_keep` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_cache_seq_keep` |
| `llama_memory_seq_add` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_cache_seq_add` |
| `llama_memory_seq_div` | ❌ MISSING | **CRITICAL** - Replaces `llama_kv_cache_seq_div` |
| `llama_memory_seq_pos_min` | ❌ MISSING | New function |
| `llama_memory_seq_pos_max` | ❌ MISSING | New function |
| `llama_memory_can_shift` | ❌ MISSING | New function |

### Old KV Cache API (REMOVE THESE)

| Function | Status | Action |
|----------|--------|--------|
| `llama_get_kv_self` | ⚡ **REMOVE** | Does not exist in llama.cpp |
| `llama_kv_self_clear` | ⚡ **REMOVE** | Does not exist in llama.cpp |
| `llama_kv_cache_*` family | ⚡ **REMOVE** | Does not exist in llama.cpp |

---

## State/Session Management

| Function | Status | Notes |
|----------|--------|-------|
| `llama_state_get_size` | ✅ DONE | Working |
| `llama_state_get_data` | ✅ DONE | Working |
| `llama_state_set_data` | ✅ DONE | Working |
| `llama_state_load_file` | ✅ DONE | Working |
| `llama_state_save_file` | ✅ DONE | Working |
| `llama_get_state_size` | 🗑️ DEPRECATED | Remove - use `llama_state_get_size` |
| `llama_copy_state_data` | 🗑️ DEPRECATED | Remove - use `llama_state_get_data` |
| `llama_set_state_data` | 🗑️ DEPRECATED | Remove - use `llama_state_set_data` |
| `llama_load_session_file` | 🗑️ DEPRECATED | Remove - use `llama_state_load_file` |
| `llama_save_session_file` | 🗑️ DEPRECATED | Remove - use `llama_state_save_file` |
| `llama_state_seq_get_size` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_get_data` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_set_data` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_save_file` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_load_file` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_get_size_ext` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_get_data_ext` | ❌ MISSING | Not in wrapper |
| `llama_state_seq_set_data_ext` | ❌ MISSING | Not in wrapper |

---

## Batch Operations

| Function | Status | Notes |
|----------|--------|-------|
| `llama_batch_get_one` | ✅ DONE | Working |
| `llama_batch_init` | ✅ DONE | Working |
| `llama_batch_free` | ✅ DONE | Working |

---

## Inference

| Function | Status | Notes |
|----------|--------|-------|
| `llama_encode` | ✅ DONE | Working |
| `llama_decode` | ✅ DONE | Working |
| `llama_set_n_threads` | ✅ DONE | Working |
| `llama_n_threads` | ❌ MISSING | Not in wrapper |
| `llama_n_threads_batch` | ❌ MISSING | Not in wrapper |
| `llama_set_embeddings` | ❌ MISSING | Not in wrapper |
| `llama_set_causal_attn` | ❌ MISSING | Not in wrapper |
| `llama_set_warmup` | ❌ MISSING | Not in wrapper |
| `llama_set_abort_callback` | ❌ MISSING | Not in wrapper |
| `llama_synchronize` | ⚠️ NEED | Exists but needs verification |

---

## Output Access

| Function | Status | Notes |
|----------|--------|-------|
| `llama_get_logits` | ✅ DONE | Working |
| `llama_get_logits_ith` | ✅ DONE | Working |
| `llama_get_embeddings` | ✅ DONE | Working |
| `llama_get_embeddings_ith` | ✅ DONE | Working |
| `llama_get_embeddings_seq` | ✅ DONE | Working |

---

## Backend Sampling (EXPERIMENTAL)

| Function | Status | Notes |
|----------|--------|-------|
| `llama_get_sampled_token_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_probs_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_probs_count_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_logits_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_logits_count_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_candidates_ith` | ❌ MISSING | Experimental feature |
| `llama_get_sampled_candidates_count_ith` | ❌ MISSING | Experimental feature |

---

## Vocabulary

| Function | Status | Notes |
|----------|--------|-------|
| `llama_vocab_type` | ✅ DONE | Working |
| `llama_vocab_n_tokens` | ✅ DONE | Working |
| `llama_vocab_get_text` | ✅ DONE | Working |
| `llama_vocab_get_score` | ✅ DONE | Working |
| `llama_vocab_get_attr` | ✅ DONE | Working |
| `llama_vocab_is_eog` | ✅ DONE | Working |
| `llama_vocab_is_control` | ✅ DONE | Working |
| `llama_vocab_bos` | ✅ DONE | Working |
| `llama_vocab_eos` | ✅ DONE | Working |
| `llama_vocab_eot` | ✅ DONE | Working |
| `llama_vocab_sep` | ✅ DONE | Working |
| `llama_vocab_nl` | ✅ DONE | Working |
| `llama_vocab_pad` | ✅ DONE | Working |
| `llama_vocab_mask` | ❌ MISSING | Not in wrapper |
| `llama_vocab_get_add_bos` | ❌ MISSING | Not in wrapper |
| `llama_vocab_get_add_eos` | ❌ MISSING | Not in wrapper |
| `llama_vocab_get_add_sep` | ❌ MISSING | Not in wrapper |
| `llama_vocab_fim_pre` | ✅ DONE | Working |
| `llama_vocab_fim_suf` | ✅ DONE | Working |
| `llama_vocab_fim_mid` | ✅ DONE | Working |
| `llama_vocab_fim_pad` | ✅ DONE | Working |
| `llama_vocab_fim_rep` | ✅ DONE | Working |
| `llama_vocab_fim_sep` | ✅ DONE | Working |
| `llama_vocab_cls` | 🗑️ DEPRECATED | Remove - use `llama_vocab_bos` |

### Deprecated Token Functions (REMOVE)

| Function | Status | Replacement |
|----------|--------|-------------|
| `llama_token_get_text` | 🗑️ DEPRECATED | `llama_vocab_get_text` |
| `llama_token_get_score` | 🗑️ DEPRECATED | `llama_vocab_get_score` |
| `llama_token_get_attr` | 🗑️ DEPRECATED | `llama_vocab_get_attr` |
| `llama_token_is_eog` | 🗑️ DEPRECATED | `llama_vocab_is_eog` |
| `llama_token_is_control` | 🗑️ DEPRECATED | `llama_vocab_is_control` |
| `llama_token_bos` | 🗑️ DEPRECATED | `llama_vocab_bos` |
| `llama_token_eos` | 🗑️ DEPRECATED | `llama_vocab_eos` |
| `llama_token_eot` | 🗑️ DEPRECATED | `llama_vocab_eot` |
| `llama_token_cls` | 🗑️ DEPRECATED | `llama_vocab_cls` |
| `llama_token_sep` | 🗑️ DEPRECATED | `llama_vocab_sep` |
| `llama_token_nl` | 🗑️ DEPRECATED | `llama_vocab_nl` |
| `llama_token_pad` | 🗑️ DEPRECATED | `llama_vocab_pad` |
| `llama_add_bos_token` | 🗑️ DEPRECATED | `llama_vocab_get_add_bos` |
| `llama_add_eos_token` | 🗑️ DEPRECATED | `llama_vocab_get_add_eos` |
| `llama_token_fim_*` | 🗑️ DEPRECATED | `llama_vocab_fim_*` |

---

## Tokenization

| Function | Status | Notes |
|----------|--------|-------|
| `llama_tokenize` | ✅ DONE | Working |
| `llama_token_to_piece` | ✅ DONE | Working |
| `llama_detokenize` | ✅ DONE | Working |

---

## Chat Templates

| Function | Status | Notes |
|----------|--------|-------|
| `llama_chat_apply_template` | ✅ DONE | Working |
| `llama_chat_builtin_templates` | ❌ MISSING | Not in wrapper |

---

## Sampling API

| Function | Status | Notes |
|----------|--------|-------|
| `llama_sampler_chain_default_params` | ✅ DONE | Working |
| `llama_set_sampler` | ❌ MISSING | Experimental feature |
| `llama_sampler_init` | ✅ DONE | Working |
| `llama_sampler_name` | ✅ DONE | Working |
| `llama_sampler_accept` | ✅ DONE | Working |
| `llama_sampler_apply` | ✅ DONE | Working |
| `llama_sampler_reset` | ✅ DONE | Working |
| `llama_sampler_clone` | ✅ DONE | Working |
| `llama_sampler_free` | ✅ DONE | Working |
| `llama_sampler_chain_init` | ✅ DONE | Working |
| `llama_sampler_chain_add` | ✅ DONE | Working |
| `llama_sampler_chain_get` | ✅ DONE | Working |
| `llama_sampler_chain_n` | ✅ DONE | Working |
| `llama_sampler_chain_remove` | ✅ DONE | Working |
| `llama_sampler_sample` | ✅ DONE | Working |
| `llama_sampler_get_seed` | ❌ MISSING | Not in wrapper |

---

## Built-in Samplers

| Function | Status | Notes |
|----------|--------|-------|
| `llama_sampler_init_greedy` | ✅ DONE | Working |
| `llama_sampler_init_dist` | ✅ DONE | Working |
| `llama_sampler_init_top_k` | ✅ DONE | Working |
| `llama_sampler_init_top_p` | ✅ DONE | Working |
| `llama_sampler_init_min_p` | ✅ DONE | Working |
| `llama_sampler_init_typical` | ✅ DONE | Working |
| `llama_sampler_init_temp` | ✅ DONE | Working |
| `llama_sampler_init_temp_ext` | ✅ DONE | Working |
| `llama_sampler_init_xtc` | ✅ DONE | Working |
| `llama_sampler_init_top_n_sigma` | ❌ MISSING | New sampler |
| `llama_sampler_init_mirostat` | ✅ DONE | Working |
| `llama_sampler_init_mirostat_v2` | ✅ DONE | Working |
| `llama_sampler_init_grammar` | ✅ DONE | Working |
| `llama_sampler_init_grammar_lazy_patterns` | ❌ MISSING | New function |
| `llama_sampler_init_penalties` | ✅ DONE | Working |
| `llama_sampler_init_dry` | ✅ DONE | Working |
| `llama_sampler_init_adaptive_p` | ❌ MISSING | New sampler |
| `llama_sampler_init_logit_bias` | ✅ DONE | Working |
| `llama_sampler_init_infill` | ✅ DONE | Working |
| `llama_sampler_init_grammar_lazy` | 🗑️ DEPRECATED | Use `llama_sampler_init_grammar_lazy_patterns` |

---

## LoRA Adapters

| Function | Status | Notes |
|----------|--------|-------|
| `llama_adapter_lora_init` | ✅ DONE | Working |
| `llama_adapter_meta_val_str` | ❌ MISSING | Not in wrapper |
| `llama_adapter_meta_count` | ❌ MISSING | Not in wrapper |
| `llama_adapter_meta_key_by_index` | ❌ MISSING | Not in wrapper |
| `llama_adapter_meta_val_str_by_index` | ❌ MISSING | Not in wrapper |
| `llama_adapter_get_alora_n_invocation_tokens` | ❌ MISSING | Not in wrapper |
| `llama_adapter_get_alora_invocation_tokens` | ❌ MISSING | Not in wrapper |
| `llama_set_adapter_lora` | ✅ DONE | Working |
| `llama_rm_adapter_lora` | ✅ DONE | Working |
| `llama_clear_adapter_lora` | ✅ DONE | Working |
| `llama_apply_adapter_cvec` | ❌ MISSING | Not in wrapper |
| `llama_adapter_lora_free` | 🗑️ DEPRECATED | Freed with model |

---

## Utilities

| Function | Status | Notes |
|----------|--------|-------|
| `llama_time_us` | ✅ DONE | Working |
| `llama_max_devices` | ✅ DONE | Working |
| `llama_max_parallel_sequences` | ✅ DONE | Working |
| `llama_max_tensor_buft_overrides` | ❌ MISSING | Not in wrapper |
| `llama_supports_mmap` | ✅ DONE | Working |
| `llama_supports_mlock` | ✅ DONE | Working |
| `llama_supports_gpu_offload` | ✅ DONE | Working |
| `llama_supports_rpc` | ✅ DONE | Working |
| `llama_split_path` | ❌ MISSING | Not in wrapper |
| `llama_split_prefix` | ❌ MISSING | Not in wrapper |
| `llama_print_system_info` | ✅ DONE | Working |
| `llama_flash_attn_type_name` | ❌ MISSING | Not in wrapper |

---

## Logging

| Function | Status | Notes |
|----------|--------|-------|
| `llama_log_get` | ❌ MISSING | Not in wrapper |
| `llama_log_set` | ✅ DONE | Working |

---

## Performance

| Function | Status | Notes |
|----------|--------|-------|
| `llama_perf_context` | ✅ DONE | Working |
| `llama_perf_context_print` | ✅ DONE | Working |
| `llama_perf_context_reset` | ✅ DONE | Working |
| `llama_perf_sampler` | ✅ DONE | Working |
| `llama_perf_sampler_print` | ✅ DONE | Working |
| `llama_perf_sampler_reset` | ✅ DONE | Working |
| `llama_memory_breakdown_print` | ❌ MISSING | Not in wrapper |

---

## Training/Optimization

| Function | Status | Notes |
|----------|--------|-------|
| `llama_opt_param_filter_all` | ❌ MISSING | Not in wrapper |
| `llama_opt_init` | ❌ MISSING | Not in wrapper |
| `llama_opt_epoch` | ❌ MISSING | Not in wrapper |
| `llama_model_quantize_default_params` | ✅ DONE | Working |

---

## Summary Statistics

### By Status

- **✅ DONE:** ~120 functions (60%)
- **❌ MISSING:** ~60 functions (30%)
- **⚠️ NEED:** ~15 functions (7.5%)
- **🗑️ DEPRECATED:** ~30 functions (15%)
- **⚡ CRITICAL:** 7 functions (blocking issues)

### Critical Path to Fix Wrapper

**Priority 0 (BLOCKING):**
1. Remove all `llama_kv_*` function calls
2. Implement `llama_memory_*` API (9 functions)
3. Remove `llama_get_kv_self()` binding

**Priority 1 (High):**
1. Remove all deprecated function bindings (~30 functions)
2. Add missing model metadata functions (10 functions)
3. Add missing context query functions (5 functions)

**Priority 2 (Medium):**
1. Add new samplers (adaptive-p, top-n-sigma)
2. Add experimental backend sampling API
3. Add missing utility functions

**Priority 3 (Low):**
1. Add training/optimization API
2. Add LoRA adapter metadata functions
3. Add state sequence management functions

---

## Migration Plan

### Phase 1: Fix Critical Blocking Issues (P0)

**Goal:** Make wrapper importable and functional

1. **Remove KV cache API calls**
   - File: `llamarcp_wrapper.py`
   - Remove: `llama_get_kv_self()` binding (line ~1449)
   - Remove: All `llama_kv_*` function calls

2. **Implement Memory API**
   - Add: `llama_memory_clear()`
   - Add: `llama_memory_seq_rm()`
   - Add: `llama_memory_seq_cp()`
   - Add: `llama_memory_seq_keep()`
   - Add: `llama_memory_seq_add()`
   - Add: `llama_memory_seq_div()`
   - Add: `llama_memory_seq_pos_min()`
   - Add: `llama_memory_seq_pos_max()`
   - Add: `llama_memory_can_shift()`

3. **Update cache management code**
   - File: `llamarcp_cache.py`
   - Replace KV cache calls with Memory API calls

**Estimated Effort:** 4-6 hours

### Phase 2: Remove Deprecated Functions (P1)

**Goal:** Clean up codebase

1. Remove ~30 deprecated function bindings
2. Update any code using deprecated functions
3. Add warnings for deprecated usage

**Estimated Effort:** 2-3 hours

### Phase 3: Add Missing Core Functions (P1)

**Goal:** Feature parity with llama.cpp

1. Add missing model metadata functions
2. Add missing context query functions
3. Add missing state management functions

**Estimated Effort:** 3-4 hours

### Phase 4: Add New Features (P2-P3)

**Goal:** Support latest llama.cpp features

1. New samplers
2. Experimental features
3. Training API

**Estimated Effort:** 6-8 hours

---

## Testing Checklist

After each phase:

- [ ] Wrapper imports without errors
- [ ] Basic model loading works
- [ ] Inference works
- [ ] Memory management works
- [ ] State save/load works
- [ ] Sampling works
- [ ] All tests pass

---

## Notes

- This document tracks ~200+ functions
- Focus on P0 issues first - wrapper is currently non-functional
- Many "DONE" functions may need verification after fixing critical issues
- Some experimental features can be skipped for initial working version
