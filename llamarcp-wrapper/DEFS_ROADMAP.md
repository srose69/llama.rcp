# llama.cpp API Functions Registry

**Generated:** 2026-01-24  
**Source:** llama.rcp experimental branch  
**Purpose:** Complete registry of all exportable C API functions for Python bindings

---

## Core API Functions

### Backend Initialization

| Function | Location | Description |
|----------|----------|-------------|
| `llama_backend_init` | `include/llama.h:426` | Initialize llama + ggml backend |
| `llama_backend_free` | `include/llama.h:429` | Free backend resources |
| `llama_numa_init` | `include/llama.h:432` | Initialize NUMA optimizations |

### Model Management

| Function | Location | Description |
|----------|----------|-------------|
| `llama_model_default_params` | `include/llama.h:418` | Get default model parameters |
| `llama_model_load_from_file` | `include/llama.h:450` | Load model from single/split GGUF file |
| `llama_model_load_from_splits` | `include/llama.h:456` | Load model from multiple split files |
| `llama_model_save_to_file` | `include/llama.h:461` | Save model to file |
| `llama_model_free` | `include/llama.h:468` | Free model resources |
| `llama_model_quantize` | `include/llama.h:613` | Quantize model to different format |

### Model Metadata

| Function | Location | Description |
|----------|----------|-------------|
| `llama_model_get_vocab` | `include/llama.h:533` | Get model vocabulary |
| `llama_model_rope_type` | `include/llama.h:534` | Get RoPE type |
| `llama_model_n_ctx_train` | `include/llama.h:536` | Get training context size |
| `llama_model_n_embd` | `include/llama.h:537` | Get embedding dimensions |
| `llama_model_n_embd_inp` | `include/llama.h:538` | Get input embedding dimensions |
| `llama_model_n_embd_out` | `include/llama.h:539` | Get output embedding dimensions |
| `llama_model_n_layer` | `include/llama.h:540` | Get number of layers |
| `llama_model_n_head` | `include/llama.h:541` | Get number of attention heads |
| `llama_model_n_head_kv` | `include/llama.h:542` | Get number of KV heads |
| `llama_model_n_swa` | `include/llama.h:543` | Get SWA parameter |
| `llama_model_rope_freq_scale_train` | `include/llama.h:546` | Get RoPE frequency scaling |
| `llama_model_n_cls_out` | `include/llama.h:550` | Get classifier output count |
| `llama_model_cls_label` | `include/llama.h:553` | Get classifier label by index |
| `llama_model_meta_val_str` | `include/llama.h:566` | Get metadata value as string |
| `llama_model_meta_count` | `include/llama.h:569` | Get metadata key/value pair count |
| `llama_model_meta_key_str` | `include/llama.h:572` | Get metadata key name |
| `llama_model_meta_key_by_index` | `include/llama.h:575` | Get metadata key by index |
| `llama_model_meta_val_str_by_index` | `include/llama.h:578` | Get metadata value by index |
| `llama_model_desc` | `include/llama.h:581` | Get model description string |
| `llama_model_size` | `include/llama.h:584` | Get total model size in bytes |
| `llama_model_chat_template` | `include/llama.h:588` | Get chat template |
| `llama_model_n_params` | `include/llama.h:591` | Get total parameter count |
| `llama_model_has_encoder` | `include/llama.h:594` | Check if model has encoder |
| `llama_model_has_decoder` | `include/llama.h:597` | Check if model has decoder |
| `llama_model_decoder_start_token` | `include/llama.h:601` | Get decoder start token |
| `llama_model_is_recurrent` | `include/llama.h:604` | Check if model is recurrent |
| `llama_model_is_hybrid` | `include/llama.h:607` | Check if model is hybrid |
| `llama_model_is_diffusion` | `include/llama.h:610` | Check if model is diffusion-based |

### Context Management

| Function | Location | Description |
|----------|----------|-------------|
| `llama_context_default_params` | `include/llama.h:419` | Get default context parameters |
| `llama_init_from_model` | `include/llama.h:470` | Initialize context from model |
| `llama_free` | `include/llama.h:480` | Free context resources |
| `llama_params_fit` | `include/llama.h:492` | Fit parameters to device memory |
| `llama_attach_threadpool` | `include/llama.h:435` | Attach custom threadpool |
| `llama_detach_threadpool` | `include/llama.h:440` | Detach threadpool |

### Context Queries

| Function | Location | Description |
|----------|----------|-------------|
| `llama_n_ctx` | `include/llama.h:516` | Get context size |
| `llama_n_ctx_seq` | `include/llama.h:517` | Get sequence context size |
| `llama_n_batch` | `include/llama.h:518` | Get batch size |
| `llama_n_ubatch` | `include/llama.h:519` | Get micro-batch size |
| `llama_n_seq_max` | `include/llama.h:520` | Get max sequences |
| `llama_get_model` | `include/llama.h:529` | Get model from context |
| `llama_get_memory` | `include/llama.h:530` | Get memory handle |
| `llama_pooling_type` | `include/llama.h:531` | Get pooling type |

### Memory Management (NEW API)

| Function | Location | Description |
|----------|----------|-------------|
| `llama_memory_clear` | `include/llama.h:694` | Clear memory contents |
| `llama_memory_seq_rm` | `include/llama.h:703` | Remove sequence tokens |
| `llama_memory_seq_cp` | `include/llama.h:712` | Copy sequence tokens |
| `llama_memory_seq_keep` | `include/llama.h:720` | Keep only specified sequence |
| `llama_memory_seq_add` | `include/llama.h:727` | Add relative position to sequence |
| `llama_memory_seq_div` | `include/llama.h:737` | Divide positions by factor |
| `llama_memory_seq_pos_min` | `include/llama.h:748` | Get minimum position in sequence |
| `llama_memory_seq_pos_max` | `include/llama.h:755` | Get maximum position in sequence |
| `llama_memory_can_shift` | `include/llama.h:760` | Check if memory supports shifting |

### State/Session Management

| Function | Location | Description |
|----------|----------|-------------|
| `llama_state_get_size` | `include/llama.h:769` | Get state size in bytes |
| `llama_state_get_data` | `include/llama.h:776` | Copy state to buffer |
| `llama_state_set_data` | `include/llama.h:787` | Restore state from buffer |
| `llama_state_load_file` | `include/llama.h:797` | Load session from file |
| `llama_state_save_file` | `include/llama.h:811` | Save session to file |
| `llama_state_seq_get_size` | `include/llama.h:824` | Get sequence state size |
| `llama_state_seq_get_data` | `include/llama.h:829` | Copy sequence state |
| `llama_state_seq_set_data` | `include/llama.h:839` | Restore sequence state |
| `llama_state_seq_save_file` | `include/llama.h:845` | Save sequence to file |
| `llama_state_seq_load_file` | `include/llama.h:852` | Load sequence from file |
| `llama_state_seq_get_size_ext` | `include/llama.h:868` | Get sequence state size (extended) |
| `llama_state_seq_get_data_ext` | `include/llama.h:873` | Copy sequence state (extended) |
| `llama_state_seq_set_data_ext` | `include/llama.h:880` | Restore sequence state (extended) |

### Batch Operations

| Function | Location | Description |
|----------|----------|-------------|
| `llama_batch_get_one` | `include/llama.h:897` | Get single-sequence batch |
| `llama_batch_init` | `include/llama.h:908` | Allocate batch on heap |
| `llama_batch_free` | `include/llama.h:914` | Free batch |

### Inference

| Function | Location | Description |
|----------|----------|-------------|
| `llama_encode` | `include/llama.h:922` | Process batch (encoder) |
| `llama_decode` | `include/llama.h:938` | Process batch (decoder) |
| `llama_set_n_threads` | `include/llama.h:945` | Set thread count |
| `llama_n_threads` | `include/llama.h:948` | Get generation thread count |
| `llama_n_threads_batch` | `include/llama.h:951` | Get batch thread count |
| `llama_set_embeddings` | `include/llama.h:955` | Enable/disable embeddings |
| `llama_set_causal_attn` | `include/llama.h:959` | Set causal attention |
| `llama_set_warmup` | `include/llama.h:963` | Set warmup mode |
| `llama_set_abort_callback` | `include/llama.h:966` | Set abort callback |
| `llama_synchronize` | `include/llama.h:971` | Wait for computations |

### Output Access

| Function | Location | Description |
|----------|----------|-------------|
| `llama_get_logits` | `include/llama.h:979` | Get all logits |
| `llama_get_logits_ith` | `include/llama.h:985` | Get logits for ith token |
| `llama_get_embeddings` | `include/llama.h:994` | Get all embeddings |
| `llama_get_embeddings_ith` | `include/llama.h:1001` | Get embeddings for ith token |
| `llama_get_embeddings_seq` | `include/llama.h:1007` | Get embeddings for sequence |

### Backend Sampling (EXPERIMENTAL)

| Function | Location | Description |
|----------|----------|-------------|
| `llama_get_sampled_token_ith` | `include/llama.h:1016` | Get backend sampled token |
| `llama_get_sampled_probs_ith` | `include/llama.h:1021` | Get sampled probabilities |
| `llama_get_sampled_probs_count_ith` | `include/llama.h:1022` | Get sampled probs count |
| `llama_get_sampled_logits_ith` | `include/llama.h:1026` | Get sampled logits |
| `llama_get_sampled_logits_count_ith` | `include/llama.h:1027` | Get sampled logits count |
| `llama_get_sampled_candidates_ith` | `include/llama.h:1032` | Get sampled candidates |
| `llama_get_sampled_candidates_count_ith` | `include/llama.h:1033` | Get sampled candidates count |

### Vocabulary

| Function | Location | Description |
|----------|----------|-------------|
| `llama_vocab_type` | `include/llama.h:555` | Get vocabulary type |
| `llama_vocab_n_tokens` | `include/llama.h:557` | Get token count |
| `llama_vocab_get_text` | `include/llama.h:1039` | Get token text |
| `llama_vocab_get_score` | `include/llama.h:1041` | Get token score |
| `llama_vocab_get_attr` | `include/llama.h:1043` | Get token attributes |
| `llama_vocab_is_eog` | `include/llama.h:1046` | Check if end-of-generation |
| `llama_vocab_is_control` | `include/llama.h:1049` | Check if control token |
| `llama_vocab_bos` | `include/llama.h:1052` | Get BOS token |
| `llama_vocab_eos` | `include/llama.h:1053` | Get EOS token |
| `llama_vocab_eot` | `include/llama.h:1054` | Get EOT token |
| `llama_vocab_sep` | `include/llama.h:1055` | Get separator token |
| `llama_vocab_nl` | `include/llama.h:1056` | Get newline token |
| `llama_vocab_pad` | `include/llama.h:1057` | Get padding token |
| `llama_vocab_mask` | `include/llama.h:1058` | Get mask token |
| `llama_vocab_get_add_bos` | `include/llama.h:1060` | Check if add BOS |
| `llama_vocab_get_add_eos` | `include/llama.h:1061` | Check if add EOS |
| `llama_vocab_get_add_sep` | `include/llama.h:1062` | Check if add separator |
| `llama_vocab_fim_pre` | `include/llama.h:1064` | Get FIM prefix token |
| `llama_vocab_fim_suf` | `include/llama.h:1065` | Get FIM suffix token |
| `llama_vocab_fim_mid` | `include/llama.h:1066` | Get FIM middle token |
| `llama_vocab_fim_pad` | `include/llama.h:1067` | Get FIM padding token |
| `llama_vocab_fim_rep` | `include/llama.h:1068` | Get FIM repeat token |
| `llama_vocab_fim_sep` | `include/llama.h:1069` | Get FIM separator token |

### Tokenization

| Function | Location | Description |
|----------|----------|-------------|
| `llama_tokenize` | `include/llama.h:1110` | Convert text to tokens |
| `llama_token_to_piece` | `include/llama.h:1124` | Convert token to text piece |
| `llama_detokenize` | `include/llama.h:1138` | Convert tokens to text |

### Chat Templates

| Function | Location | Description |
|----------|----------|-------------|
| `llama_chat_apply_template` | `include/llama.h:1161` | Apply chat template |
| `llama_chat_builtin_templates` | `include/llama.h:1170` | Get built-in templates |

### Sampling API

| Function | Location | Description |
|----------|----------|-------------|
| `llama_sampler_chain_default_params` | `include/llama.h:420` | Get default sampler params |
| `llama_set_sampler` | `include/llama.h:1259` | Attach sampler to context |
| `llama_sampler_init` | `include/llama.h:1262` | Initialize custom sampler |
| `llama_sampler_name` | `include/llama.h:1263` | Get sampler name |
| `llama_sampler_accept` | `include/llama.h:1264` | Accept token |
| `llama_sampler_apply` | `include/llama.h:1265` | Apply sampler |
| `llama_sampler_reset` | `include/llama.h:1266` | Reset sampler |
| `llama_sampler_clone` | `include/llama.h:1267` | Clone sampler |
| `llama_sampler_free` | `include/llama.h:1269` | Free sampler |
| `llama_sampler_chain_init` | `include/llama.h:1274` | Initialize sampler chain |
| `llama_sampler_chain_add` | `include/llama.h:1277` | Add sampler to chain |
| `llama_sampler_chain_get` | `include/llama.h:1284` | Get sampler from chain |
| `llama_sampler_chain_n` | `include/llama.h:1287` | Get chain length |
| `llama_sampler_chain_remove` | `include/llama.h:1290` | Remove sampler from chain |
| `llama_sampler_sample` | `include/llama.h:1466` | Sample token |
| `llama_sampler_get_seed` | `include/llama.h:1454` | Get sampler seed |

### Built-in Samplers

| Function | Location | Description |
|----------|----------|-------------|
| `llama_sampler_init_greedy` | `include/llama.h:1294` | Greedy sampler |
| `llama_sampler_init_dist` | `include/llama.h:1297` | Distribution sampler |
| `llama_sampler_init_top_k` | `include/llama.h:1301` | Top-K sampler |
| `llama_sampler_init_top_p` | `include/llama.h:1304` | Top-P (nucleus) sampler |
| `llama_sampler_init_min_p` | `include/llama.h:1307` | Min-P sampler |
| `llama_sampler_init_typical` | `include/llama.h:1310` | Typical sampler |
| `llama_sampler_init_temp` | `include/llama.h:1313` | Temperature sampler |
| `llama_sampler_init_temp_ext` | `include/llama.h:1316` | Dynamic temperature |
| `llama_sampler_init_xtc` | `include/llama.h:1319` | XTC sampler |
| `llama_sampler_init_top_n_sigma` | `include/llama.h:1322` | Top-n-sigma sampler |
| `llama_sampler_init_mirostat` | `include/llama.h:1330` | Mirostat 1.0 |
| `llama_sampler_init_mirostat_v2` | `include/llama.h:1342` | Mirostat 2.0 |
| `llama_sampler_init_grammar` | `include/llama.h:1351` | Grammar sampler |
| `llama_sampler_init_grammar_lazy_patterns` | `include/llama.h:1370` | Lazy grammar sampler |
| `llama_sampler_init_penalties` | `include/llama.h:1381` | Repetition penalties |
| `llama_sampler_init_dry` | `include/llama.h:1388` | DRY sampler |
| `llama_sampler_init_adaptive_p` | `include/llama.h:1420` | Adaptive-P sampler |
| `llama_sampler_init_logit_bias` | `include/llama.h:1425` | Logit bias sampler |
| `llama_sampler_init_infill` | `include/llama.h:1451` | Infill sampler |

### LoRA Adapters

| Function | Location | Description |
|----------|----------|-------------|
| `llama_adapter_lora_init` | `include/llama.h:625` | Load LoRA adapter |
| `llama_adapter_meta_val_str` | `include/llama.h:636` | Get adapter metadata value |
| `llama_adapter_meta_count` | `include/llama.h:639` | Get adapter metadata count |
| `llama_adapter_meta_key_by_index` | `include/llama.h:642` | Get adapter metadata key |
| `llama_adapter_meta_val_str_by_index` | `include/llama.h:645` | Get adapter metadata value by index |
| `llama_adapter_get_alora_n_invocation_tokens` | `include/llama.h:653` | Get ALoRA invocation token count |
| `llama_adapter_get_alora_invocation_tokens` | `include/llama.h:654` | Get ALoRA invocation tokens |
| `llama_set_adapter_lora` | `include/llama.h:660` | Add LoRA to context |
| `llama_rm_adapter_lora` | `include/llama.h:667` | Remove LoRA from context |
| `llama_clear_adapter_lora` | `include/llama.h:672` | Clear all LoRAs |
| `llama_apply_adapter_cvec` | `include/llama.h:680` | Apply control vector |

### Utilities

| Function | Location | Description |
|----------|----------|-------------|
| `llama_time_us` | `include/llama.h:502` | Get time in microseconds |
| `llama_max_devices` | `include/llama.h:504` | Get max device count |
| `llama_max_parallel_sequences` | `include/llama.h:505` | Get max parallel sequences |
| `llama_max_tensor_buft_overrides` | `include/llama.h:506` | Get max tensor buffer overrides |
| `llama_supports_mmap` | `include/llama.h:508` | Check mmap support |
| `llama_supports_mlock` | `include/llama.h:509` | Check mlock support |
| `llama_supports_gpu_offload` | `include/llama.h:510` | Check GPU offload support |
| `llama_supports_rpc` | `include/llama.h:511` | Check RPC support |
| `llama_split_path` | `include/llama.h:1478` | Build split GGUF path |
| `llama_split_prefix` | `include/llama.h:1483` | Extract split path prefix |
| `llama_print_system_info` | `include/llama.h:1486` | Get system info string |
| `llama_flash_attn_type_name` | `include/llama.h:189` | Get flash attention type name |

### Logging

| Function | Location | Description |
|----------|----------|-------------|
| `llama_log_get` | `include/llama.h:1491` | Get current log callback |
| `llama_log_set` | `include/llama.h:1492` | Set log callback |

### Performance

| Function | Location | Description |
|----------|----------|-------------|
| `llama_perf_context` | `include/llama.h:1518` | Get context performance data |
| `llama_perf_context_print` | `include/llama.h:1519` | Print context performance |
| `llama_perf_context_reset` | `include/llama.h:1520` | Reset context performance |
| `llama_perf_sampler` | `include/llama.h:1523` | Get sampler performance data |
| `llama_perf_sampler_print` | `include/llama.h:1524` | Print sampler performance |
| `llama_perf_sampler_reset` | `include/llama.h:1525` | Reset sampler performance |
| `llama_memory_breakdown_print` | `include/llama.h:1528` | Print memory breakdown |

### Training/Optimization

| Function | Location | Description |
|----------|----------|-------------|
| `llama_opt_param_filter_all` | `include/llama.h:1538` | Filter all trainable params |
| `llama_opt_init` | `include/llama.h:1552` | Initialize optimizer |
| `llama_opt_epoch` | `include/llama.h:1554` | Run training epoch |
| `llama_model_quantize_default_params` | `include/llama.h:421` | Get default quantize params |

---

## Deprecated Functions (DO NOT USE)

These functions are marked as deprecated and should be replaced with their modern equivalents:

| Deprecated Function | Replacement | Location |
|---------------------|-------------|----------|
| `llama_load_model_from_file` | `llama_model_load_from_file` | `include/llama.h:442` |
| `llama_free_model` | `llama_model_free` | `include/llama.h:465` |
| `llama_new_context_with_model` | `llama_init_from_model` | `include/llama.h:474` |
| `llama_get_state_size` | `llama_state_get_size` | `include/llama.h:770` |
| `llama_copy_state_data` | `llama_state_get_data` | `include/llama.h:780` |
| `llama_set_state_data` | `llama_state_set_data` | `include/llama.h:791` |
| `llama_load_session_file` | `llama_state_load_file` | `include/llama.h:803` |
| `llama_save_session_file` | `llama_state_save_file` | `include/llama.h:816` |
| `llama_n_ctx_train` | `llama_model_n_ctx_train` | `include/llama.h:522` |
| `llama_n_embd` | `llama_model_n_embd` | `include/llama.h:523` |
| `llama_n_layer` | `llama_model_n_layer` | `include/llama.h:524` |
| `llama_n_head` | `llama_model_n_head` | `include/llama.h:525` |
| `llama_n_vocab` | `llama_vocab_n_tokens` | `include/llama.h:527` |
| `llama_vocab_cls` | `llama_vocab_bos` | `include/llama.h:1093` |
| `llama_token_get_text` | `llama_vocab_get_text` | `include/llama.h:1071` |
| `llama_token_get_score` | `llama_vocab_get_score` | `include/llama.h:1072` |
| `llama_token_get_attr` | `llama_vocab_get_attr` | `include/llama.h:1073` |
| `llama_token_is_eog` | `llama_vocab_is_eog` | `include/llama.h:1074` |
| `llama_token_is_control` | `llama_vocab_is_control` | `include/llama.h:1075` |
| `llama_token_bos` | `llama_vocab_bos` | `include/llama.h:1076` |
| `llama_token_eos` | `llama_vocab_eos` | `include/llama.h:1077` |
| `llama_token_eot` | `llama_vocab_eot` | `include/llama.h:1078` |
| `llama_token_cls` | `llama_vocab_cls` | `include/llama.h:1079` |
| `llama_token_sep` | `llama_vocab_sep` | `include/llama.h:1080` |
| `llama_token_nl` | `llama_vocab_nl` | `include/llama.h:1081` |
| `llama_token_pad` | `llama_vocab_pad` | `include/llama.h:1082` |
| `llama_add_bos_token` | `llama_vocab_get_add_bos` | `include/llama.h:1083` |
| `llama_add_eos_token` | `llama_vocab_get_add_eos` | `include/llama.h:1084` |
| `llama_token_fim_pre` | `llama_vocab_fim_pre` | `include/llama.h:1085` |
| `llama_token_fim_suf` | `llama_vocab_fim_suf` | `include/llama.h:1086` |
| `llama_token_fim_mid` | `llama_vocab_fim_mid` | `include/llama.h:1087` |
| `llama_token_fim_pad` | `llama_vocab_fim_pad` | `include/llama.h:1088` |
| `llama_token_fim_rep` | `llama_vocab_fim_rep` | `include/llama.h:1089` |
| `llama_token_fim_sep` | `llama_vocab_fim_sep` | `include/llama.h:1090` |
| `llama_adapter_lora_free` | (freed with model) | `include/llama.h:649` |
| `llama_sampler_init_grammar_lazy` | `llama_sampler_init_grammar_lazy_patterns` | `include/llama.h:1356` |

---

## Critical API Changes

### Removed Functions (NOT AVAILABLE)

These functions were present in older llama.cpp versions but have been **completely removed**:

- `llama_get_kv_self()` - **REMOVED** - Use new memory API instead
- `llama_kv_self_clear()` - **REMOVED** - Use `llama_memory_clear()` instead
- `llama_kv_cache_*()` family - **REMOVED** - Use `llama_memory_*()` family instead

### New Memory API (Replacement for KV Cache API)

The old KV cache API has been replaced with a unified memory API:

| Old Function | New Function | Notes |
|--------------|--------------|-------|
| `llama_kv_self_clear()` | `llama_memory_clear()` | Clear all memory |
| `llama_kv_cache_seq_rm()` | `llama_memory_seq_rm()` | Remove sequence |
| `llama_kv_cache_seq_cp()` | `llama_memory_seq_cp()` | Copy sequence |
| `llama_kv_cache_seq_keep()` | `llama_memory_seq_keep()` | Keep sequence |
| `llama_kv_cache_seq_add()` | `llama_memory_seq_add()` | Add position offset |
| `llama_kv_cache_seq_div()` | `llama_memory_seq_div()` | Divide positions |

---

## Python Wrapper Migration Guide

### Priority 1: Critical Replacements

1. Replace all `llama_get_kv_self()` calls with `llama_get_memory()`
2. Replace all `llama_kv_*()` calls with `llama_memory_*()`
3. Update deprecated function calls to modern equivalents

### Priority 2: New Features to Expose

1. Memory management API (`llama_memory_*`)
2. Extended state management (`llama_state_seq_*_ext`)
3. Backend sampling API (experimental)
4. New samplers (adaptive-p, top-n-sigma, etc.)

### Priority 3: Cleanup

1. Remove all deprecated function bindings
2. Add type hints for new functions
3. Update documentation

---

## Notes

- All functions marked `LLAMA_API` are exported and can be bound to Python
- Functions marked `DEPRECATED` should not be used in new code
- Experimental features may change in future versions
- Memory API is the new standard - KV cache API is obsolete

**Total Functions:** ~200+ exportable API functions  
**Deprecated:** ~30+ functions  
**Removed:** ~10+ functions (KV cache API)
