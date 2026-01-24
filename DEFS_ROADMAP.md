# LLAMA.CPP API DEFINITIONS ROADMAP

**Generated**: 2026-01-24

## Overview

Complete mapping of all exportable functions from llama.cpp C API and their Python bindings.

**Statistics:**
- C API Functions (llama.h): 233
- Python Wrapper Functions (llamarcp_wrapper.py): 219

---

## C API Functions (include/llama.h)

### `llama_flash_attn_type_name`
**Location:** `include/llama.h:189`

### `llama_model_params`
**Location:** `include/llama.h:418`

### `llama_context_params`
**Location:** `include/llama.h:419`

### `llama_sampler_chain_params`
**Location:** `include/llama.h:420`

### `llama_model_quantize_params`
**Location:** `include/llama.h:421`

### `llama_backend_init`
**Location:** `include/llama.h:426`

### `llama_backend_free`
**Location:** `include/llama.h:429`

### `llama_numa_init`
**Location:** `include/llama.h:432`

### `llama_attach_threadpool`
**Location:** `include/llama.h:435`

### `llama_detach_threadpool`
**Location:** `include/llama.h:440`

### `llama_model`
**Location:** `include/llama.h:442`

### `llama_model`
**Location:** `include/llama.h:450`

### `llama_model`
**Location:** `include/llama.h:456`

### `llama_model_save_to_file`
**Location:** `include/llama.h:461`

### `llama_free_model`
**Location:** `include/llama.h:465`

### `llama_model_free`
**Location:** `include/llama.h:468`

### `llama_context`
**Location:** `include/llama.h:470`

### `llama_context`
**Location:** `include/llama.h:474`

### `llama_free`
**Location:** `include/llama.h:480`

### `llama_params_fit_status`
**Location:** `include/llama.h:492`

### `llama_time_us`
**Location:** `include/llama.h:502`

### `llama_max_devices`
**Location:** `include/llama.h:504`

### `llama_max_parallel_sequences`
**Location:** `include/llama.h:505`

### `llama_max_tensor_buft_overrides`
**Location:** `include/llama.h:506`

### `llama_supports_mmap`
**Location:** `include/llama.h:508`

### `llama_supports_mlock`
**Location:** `include/llama.h:509`

### `llama_supports_gpu_offload`
**Location:** `include/llama.h:510`

### `llama_supports_rpc`
**Location:** `include/llama.h:511`

### `llama_n_ctx`
**Location:** `include/llama.h:516`

### `llama_n_ctx_seq`
**Location:** `include/llama.h:517`

### `llama_n_batch`
**Location:** `include/llama.h:518`

### `llama_n_ubatch`
**Location:** `include/llama.h:519`

### `llama_n_seq_max`
**Location:** `include/llama.h:520`

### `llama_n_ctx_train`
**Location:** `include/llama.h:522`

### `llama_n_embd`
**Location:** `include/llama.h:523`

### `llama_n_layer`
**Location:** `include/llama.h:524`

### `llama_n_head`
**Location:** `include/llama.h:525`

### `llama_n_vocab`
**Location:** `include/llama.h:527`

### `llama_model`
**Location:** `include/llama.h:529`

### `llama_memory_t`
**Location:** `include/llama.h:530`

### `llama_pooling_type`
**Location:** `include/llama.h:531`

### `llama_vocab`
**Location:** `include/llama.h:533`

### `llama_rope_type`
**Location:** `include/llama.h:534`

### `llama_model_n_ctx_train`
**Location:** `include/llama.h:536`

### `llama_model_n_embd`
**Location:** `include/llama.h:537`

### `llama_model_n_embd_inp`
**Location:** `include/llama.h:538`

### `llama_model_n_embd_out`
**Location:** `include/llama.h:539`

### `llama_model_n_layer`
**Location:** `include/llama.h:540`

### `llama_model_n_head`
**Location:** `include/llama.h:541`

### `llama_model_n_head_kv`
**Location:** `include/llama.h:542`

### `llama_model_n_swa`
**Location:** `include/llama.h:543`

### `llama_model_rope_freq_scale_train`
**Location:** `include/llama.h:546`

### `llama_model_n_cls_out`
**Location:** `include/llama.h:550`

### `llama_model_cls_label`
**Location:** `include/llama.h:553`

### `llama_vocab_type`
**Location:** `include/llama.h:555`

### `llama_vocab_n_tokens`
**Location:** `include/llama.h:557`

### `llama_model_meta_val_str`
**Location:** `include/llama.h:566`

### `llama_model_meta_count`
**Location:** `include/llama.h:569`

### `llama_model_meta_key_str`
**Location:** `include/llama.h:572`

### `llama_model_meta_key_by_index`
**Location:** `include/llama.h:575`

### `llama_model_meta_val_str_by_index`
**Location:** `include/llama.h:578`

### `llama_model_desc`
**Location:** `include/llama.h:581`

### `llama_model_size`
**Location:** `include/llama.h:584`

### `llama_model_chat_template`
**Location:** `include/llama.h:588`

### `llama_model_n_params`
**Location:** `include/llama.h:591`

### `llama_model_has_encoder`
**Location:** `include/llama.h:594`

### `llama_model_has_decoder`
**Location:** `include/llama.h:597`

### `llama_token`
**Location:** `include/llama.h:601`

### `llama_model_is_recurrent`
**Location:** `include/llama.h:604`

### `llama_model_is_hybrid`
**Location:** `include/llama.h:607`

### `llama_model_is_diffusion`
**Location:** `include/llama.h:610`

### `llama_model_quantize`
**Location:** `include/llama.h:613`

### `llama_adapter_lora`
**Location:** `include/llama.h:625`

### `llama_adapter_meta_val_str`
**Location:** `include/llama.h:636`

### `llama_adapter_meta_count`
**Location:** `include/llama.h:639`

### `llama_adapter_meta_key_by_index`
**Location:** `include/llama.h:642`

### `llama_adapter_meta_val_str_by_index`
**Location:** `include/llama.h:645`

### `llama_adapter_lora_free`
**Location:** `include/llama.h:649`

### `llama_adapter_get_alora_n_invocation_tokens`
**Location:** `include/llama.h:653`

### `llama_token`
**Location:** `include/llama.h:654`

### `llama_set_adapter_lora`
**Location:** `include/llama.h:660`

### `llama_rm_adapter_lora`
**Location:** `include/llama.h:667`

### `llama_clear_adapter_lora`
**Location:** `include/llama.h:672`

### `llama_apply_adapter_cvec`
**Location:** `include/llama.h:680`

### `llama_memory_clear`
**Location:** `include/llama.h:694`

### `llama_memory_seq_rm`
**Location:** `include/llama.h:703`

### `llama_memory_seq_cp`
**Location:** `include/llama.h:712`

### `llama_memory_seq_keep`
**Location:** `include/llama.h:720`

### `llama_memory_seq_add`
**Location:** `include/llama.h:727`

### `llama_memory_seq_div`
**Location:** `include/llama.h:737`

### `llama_pos`
**Location:** `include/llama.h:748`

### `llama_pos`
**Location:** `include/llama.h:755`

### `llama_memory_can_shift`
**Location:** `include/llama.h:760`

### `llama_state_get_size`
**Location:** `include/llama.h:769`

### `llama_get_state_size`
**Location:** `include/llama.h:770`

### `llama_state_get_data`
**Location:** `include/llama.h:776`

### `llama_copy_state_data`
**Location:** `include/llama.h:780`

### `llama_state_set_data`
**Location:** `include/llama.h:787`

### `llama_set_state_data`
**Location:** `include/llama.h:791`

### `llama_state_load_file`
**Location:** `include/llama.h:797`

### `llama_load_session_file`
**Location:** `include/llama.h:803`

### `llama_state_save_file`
**Location:** `include/llama.h:811`

### `llama_save_session_file`
**Location:** `include/llama.h:816`

### `llama_state_seq_get_size`
**Location:** `include/llama.h:824`

### `llama_state_seq_get_data`
**Location:** `include/llama.h:829`

### `llama_state_seq_set_data`
**Location:** `include/llama.h:839`

### `llama_state_seq_save_file`
**Location:** `include/llama.h:845`

### `llama_state_seq_load_file`
**Location:** `include/llama.h:852`

### `llama_state_seq_get_size_ext`
**Location:** `include/llama.h:868`

### `llama_state_seq_get_data_ext`
**Location:** `include/llama.h:873`

### `llama_state_seq_set_data_ext`
**Location:** `include/llama.h:880`

### `llama_batch`
**Location:** `include/llama.h:897`

### `llama_batch`
**Location:** `include/llama.h:908`

### `llama_batch_free`
**Location:** `include/llama.h:914`

### `llama_encode`
**Location:** `include/llama.h:922`

### `llama_decode`
**Location:** `include/llama.h:938`

### `llama_set_n_threads`
**Location:** `include/llama.h:945`

### `llama_n_threads`
**Location:** `include/llama.h:948`

### `llama_n_threads_batch`
**Location:** `include/llama.h:951`

### `llama_set_embeddings`
**Location:** `include/llama.h:955`

### `llama_set_causal_attn`
**Location:** `include/llama.h:959`

### `llama_set_warmup`
**Location:** `include/llama.h:963`

### `llama_set_abort_callback`
**Location:** `include/llama.h:966`

### `llama_synchronize`
**Location:** `include/llama.h:971`

### `llama_get_logits`
**Location:** `include/llama.h:979`

### `llama_get_logits_ith`
**Location:** `include/llama.h:985`

### `llama_get_embeddings`
**Location:** `include/llama.h:994`

### `llama_get_embeddings_ith`
**Location:** `include/llama.h:1001`

### `llama_get_embeddings_seq`
**Location:** `include/llama.h:1007`

### `llama_token`
**Location:** `include/llama.h:1016`

### `llama_get_sampled_probs_ith`
**Location:** `include/llama.h:1021`

### `llama_get_sampled_probs_count_ith`
**Location:** `include/llama.h:1022`

### `llama_get_sampled_logits_ith`
**Location:** `include/llama.h:1026`

### `llama_get_sampled_logits_count_ith`
**Location:** `include/llama.h:1027`

### `llama_token`
**Location:** `include/llama.h:1032`

### `llama_get_sampled_candidates_count_ith`
**Location:** `include/llama.h:1033`

### `llama_vocab_get_text`
**Location:** `include/llama.h:1039`

### `llama_vocab_get_score`
**Location:** `include/llama.h:1041`

### `llama_token_attr`
**Location:** `include/llama.h:1043`

### `llama_vocab_is_eog`
**Location:** `include/llama.h:1046`

### `llama_vocab_is_control`
**Location:** `include/llama.h:1049`

### `llama_token`
**Location:** `include/llama.h:1052`

### `llama_token`
**Location:** `include/llama.h:1053`

### `llama_token`
**Location:** `include/llama.h:1054`

### `llama_token`
**Location:** `include/llama.h:1055`

### `llama_token`
**Location:** `include/llama.h:1056`

### `llama_token`
**Location:** `include/llama.h:1057`

### `llama_token`
**Location:** `include/llama.h:1058`

### `llama_vocab_get_add_bos`
**Location:** `include/llama.h:1060`

### `llama_vocab_get_add_eos`
**Location:** `include/llama.h:1061`

### `llama_vocab_get_add_sep`
**Location:** `include/llama.h:1062`

### `llama_token`
**Location:** `include/llama.h:1064`

### `llama_token`
**Location:** `include/llama.h:1065`

### `llama_token`
**Location:** `include/llama.h:1066`

### `llama_token`
**Location:** `include/llama.h:1067`

### `llama_token`
**Location:** `include/llama.h:1068`

### `llama_token`
**Location:** `include/llama.h:1069`

### `llama_token_get_text`
**Location:** `include/llama.h:1071`

### `llama_token_get_score`
**Location:** `include/llama.h:1072`

### `llama_token_attr`
**Location:** `include/llama.h:1073`

### `llama_token_is_eog`
**Location:** `include/llama.h:1074`

### `llama_token_is_control`
**Location:** `include/llama.h:1075`

### `llama_token`
**Location:** `include/llama.h:1076`

### `llama_token`
**Location:** `include/llama.h:1077`

### `llama_token`
**Location:** `include/llama.h:1078`

### `llama_token`
**Location:** `include/llama.h:1079`

### `llama_token`
**Location:** `include/llama.h:1080`

### `llama_token`
**Location:** `include/llama.h:1081`

### `llama_token`
**Location:** `include/llama.h:1082`

### `llama_add_bos_token`
**Location:** `include/llama.h:1083`

### `llama_add_eos_token`
**Location:** `include/llama.h:1084`

### `llama_token`
**Location:** `include/llama.h:1085`

### `llama_token`
**Location:** `include/llama.h:1086`

### `llama_token`
**Location:** `include/llama.h:1087`

### `llama_token`
**Location:** `include/llama.h:1088`

### `llama_token`
**Location:** `include/llama.h:1089`

### `llama_token`
**Location:** `include/llama.h:1090`

### `llama_token`
**Location:** `include/llama.h:1093`

### `llama_tokenize`
**Location:** `include/llama.h:1110`

### `llama_token_to_piece`
**Location:** `include/llama.h:1124`

### `llama_detokenize`
**Location:** `include/llama.h:1138`

### `llama_chat_apply_template`
**Location:** `include/llama.h:1161`

### `llama_chat_builtin_templates`
**Location:** `include/llama.h:1170`

### `llama_set_sampler`
**Location:** `include/llama.h:1259`

### `llama_sampler`
**Location:** `include/llama.h:1262`

### `llama_sampler_name`
**Location:** `include/llama.h:1263`

### `llama_sampler_accept`
**Location:** `include/llama.h:1264`

### `llama_sampler_apply`
**Location:** `include/llama.h:1265`

### `llama_sampler_reset`
**Location:** `include/llama.h:1266`

### `llama_sampler`
**Location:** `include/llama.h:1267`

### `llama_sampler_free`
**Location:** `include/llama.h:1269`

### `llama_sampler`
**Location:** `include/llama.h:1274`

### `llama_sampler_chain_add`
**Location:** `include/llama.h:1277`

### `llama_sampler`
**Location:** `include/llama.h:1284`

### `llama_sampler_chain_n`
**Location:** `include/llama.h:1287`

### `llama_sampler`
**Location:** `include/llama.h:1290`

### `llama_sampler`
**Location:** `include/llama.h:1294`

### `llama_sampler`
**Location:** `include/llama.h:1297`

### `llama_sampler`
**Location:** `include/llama.h:1301`

### `llama_sampler`
**Location:** `include/llama.h:1304`

### `llama_sampler`
**Location:** `include/llama.h:1307`

### `llama_sampler`
**Location:** `include/llama.h:1310`

### `llama_sampler`
**Location:** `include/llama.h:1313`

### `llama_sampler`
**Location:** `include/llama.h:1316`

### `llama_sampler`
**Location:** `include/llama.h:1319`

### `llama_sampler`
**Location:** `include/llama.h:1322`

### `llama_sampler`
**Location:** `include/llama.h:1330`

### `llama_sampler`
**Location:** `include/llama.h:1342`

### `llama_sampler`
**Location:** `include/llama.h:1351`

### `llama_sampler`
**Location:** `include/llama.h:1356`

### `llama_sampler`
**Location:** `include/llama.h:1370`

### `llama_sampler`
**Location:** `include/llama.h:1381`

### `llama_sampler`
**Location:** `include/llama.h:1388`

### `llama_sampler`
**Location:** `include/llama.h:1420`

### `llama_sampler`
**Location:** `include/llama.h:1425`

### `llama_sampler`
**Location:** `include/llama.h:1451`

### `llama_sampler_get_seed`
**Location:** `include/llama.h:1454`

### `llama_token`
**Location:** `include/llama.h:1466`

### `llama_split_path`
**Location:** `include/llama.h:1478`

### `llama_split_prefix`
**Location:** `include/llama.h:1483`

### `llama_print_system_info`
**Location:** `include/llama.h:1486`

### `llama_log_get`
**Location:** `include/llama.h:1491`

### `llama_log_set`
**Location:** `include/llama.h:1492`

### `llama_perf_context_data`
**Location:** `include/llama.h:1518`

### `llama_perf_context_print`
**Location:** `include/llama.h:1519`

### `llama_perf_context_reset`
**Location:** `include/llama.h:1520`

### `llama_perf_sampler_data`
**Location:** `include/llama.h:1523`

### `llama_perf_sampler_print`
**Location:** `include/llama.h:1524`

### `llama_perf_sampler_reset`
**Location:** `include/llama.h:1525`

### `llama_memory_breakdown_print`
**Location:** `include/llama.h:1528`

### `llama_opt_param_filter_all`
**Location:** `include/llama.h:1538`

### `llama_opt_init`
**Location:** `include/llama.h:1552`

### `llama_opt_epoch`
**Location:** `include/llama.h:1554`

---

## Python Wrapper Functions (llamarcp-wrapper/llamarcp/llamarcp_wrapper.py)

### `llama_model_default_params`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1087`

### `llama_context_default_params`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1098`

### `llama_sampler_chain_default_params`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1109`

### `llama_model_quantize_default_params`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1120`

### `llama_backend_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1134`

### `llama_backend_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1164`

### `llama_numa_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1176`

### `llama_load_model_from_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1201`

### `llama_model_load_from_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1218`

### `llama_model_load_from_splits`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1240`

### `llama_model_save_to_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1257`

### `llama_free_model`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1269`

### `llama_model_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1279`

### `llama_init_from_model`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1291`

### `llama_new_context_with_model`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1306`

### `llama_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1319`

### `llama_time_us`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1330`

### `llama_max_devices`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1336`

### `llama_max_parallel_sequences`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1342`

### `llama_supports_mmap`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1348`

### `llama_supports_mlock`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1354`

### `llama_supports_gpu_offload`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1360`

### `llama_supports_rpc`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1366`

### `llama_n_ctx`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1372`

### `llama_n_batch`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1378`

### `llama_n_ubatch`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1384`

### `llama_n_seq_max`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1390`

### `llama_n_ctx_train`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1396`

### `llama_n_embd`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1402`

### `llama_n_layer`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1408`

### `llama_n_head`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1414`

### `llama_n_vocab`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1420`

### `llama_get_model`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1426`

### `llama_get_memory`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1432`

### `llama_pooling_type`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1439`

### `llama_get_kv_self`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1449`

### `llama_model_get_vocab`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1456`

### `llama_model_rope_type`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1462`

### `llama_model_n_ctx_train`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1468`

### `llama_model_n_embd`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1474`

### `llama_model_n_layer`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1480`

### `llama_model_n_head`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1486`

### `llama_model_n_head_kv`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1492`

### `llama_model_n_swa`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1498`

### `llama_model_rope_freq_scale_train`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1505`

### `llama_model_n_cls_out`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1513`

### `llama_model_cls_label`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1521`

### `llama_vocab_type`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1528`

### `llama_vocab_n_tokens`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1534`

### `llama_model_meta_val_str`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1557`

### `llama_model_meta_count`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1571`

### `llama_model_meta_key_by_index`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1588`

### `llama_model_meta_val_str_by_index`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1611`

### `llama_model_desc`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1629`

### `llama_flash_attn_type_name`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1645`

### `llama_model_size`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1653`

### `llama_model_chat_template`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1662`

### `llama_model_n_params`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1671`

### `llama_model_has_encoder`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1679`

### `llama_model_has_decoder`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1687`

### `llama_model_decoder_start_token`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1698`

### `llama_model_is_recurrent`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1708`

### `llama_model_is_diffusion`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1716`

### `llama_model_quantize`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1735`

### `llama_adapter_lora_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1758`

### `llama_adapter_lora_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1772`

### `llama_set_adapter_lora`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1790`

### `llama_rm_adapter_lora`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1808`

### `llama_clear_adapter_lora`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1823`

### `llama_apply_adapter_cvec`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1853`

### `llama_memory_clear`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1885`

### `llama_memory_seq_rm`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1911`

### `llama_memory_seq_cp`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1948`

### `llama_memory_seq_keep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1969`

### `llama_memory_seq_add`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:1994`

### `llama_memory_seq_div`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2028`

### `llama_memory_seq_pos_min`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2052`

### `llama_memory_seq_pos_max`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2070`

### `llama_memory_can_shift`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2081`

### `llama_kv_self_n_tokens`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2097`

### `llama_kv_self_used_cells`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2108`

### `llama_kv_self_clear`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2120`

### `llama_kv_self_seq_rm`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2146`

### `llama_kv_self_seq_cp`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2179`

### `llama_kv_self_seq_keep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2199`

### `llama_kv_self_seq_add`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2227`

### `llama_kv_self_seq_div`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2262`

### `llama_kv_self_seq_pos_min`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2285`

### `llama_kv_self_seq_pos_max`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2302`

### `llama_kv_self_defrag`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2315`

### `llama_kv_self_can_shift`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2324`

### `llama_kv_self_update`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2333`

### `llama_state_get_size`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2347`

### `llama_get_state_size`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2355`

### `llama_state_get_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2376`

### `llama_copy_state_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2400`

### `llama_state_set_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2418`

### `llama_set_state_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2438`

### `llama_state_load_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2463`

### `llama_load_session_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2492`

### `llama_state_save_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2518`

### `llama_save_session_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2544`

### `llama_state_seq_get_size`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2563`

### `llama_state_seq_get_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2584`

### `llama_state_seq_set_data`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2614`

### `llama_state_seq_save_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2642`

### `llama_state_seq_load_file`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2672`

### `llama_batch_get_one`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2705`

### `llama_batch_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2731`

### `llama_batch_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2750`

### `llama_encode`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2765`

### `llama_decode`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2788`

### `llama_set_n_threads`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2811`

### `llama_n_threads`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2827`

### `llama_n_threads_batch`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2835`

### `llama_set_embeddings`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2844`

### `llama_set_causal_attn`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2853`

### `llama_set_warmup`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2863`

### `llama_set_abort_callback`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2876`

### `llama_synchronize`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2891`

### `llama_get_logits`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2908`

### `llama_get_logits_ith`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2930`

### `llama_get_embeddings`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2949`

### `llama_get_embeddings_ith`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2966`

### `llama_get_embeddings_seq`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:2984`

### `llama_vocab_get_text`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3001`

### `llama_vocab_get_score`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3011`

### `llama_vocab_get_attr`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3021`

### `llama_vocab_is_eog`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3032`

### `llama_vocab_is_control`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3042`

### `llama_vocab_bos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3052`

### `llama_vocab_eos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3059`

### `llama_vocab_eot`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3066`

### `llama_vocab_sep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3073`

### `llama_vocab_nl`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3080`

### `llama_vocab_pad`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3087`

### `llama_vocab_mask`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3094`

### `llama_vocab_get_add_bos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3105`

### `llama_vocab_get_add_eos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3115`

### `llama_vocab_get_add_sep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3125`

### `llama_vocab_fim_pre`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3135`

### `llama_vocab_fim_suf`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3145`

### `llama_vocab_fim_mid`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3155`

### `llama_vocab_fim_pad`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3165`

### `llama_vocab_fim_rep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3175`

### `llama_vocab_fim_sep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3185`

### `llama_token_get_text`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3196`

### `llama_token_get_score`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3208`

### `llama_token_get_attr`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3219`

### `llama_token_is_eog`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3230`

### `llama_token_is_control`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3241`

### `llama_token_bos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3252`

### `llama_token_eos`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3261`

### `llama_token_eot`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3270`

### `llama_token_cls`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3279`

### `llama_token_sep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3288`

### `llama_token_nl`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3298`

### `llama_token_pad`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3308`

### `llama_add_bos_token`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3318`

### `llama_add_eos_token`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3327`

### `llama_token_fim_pre`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3337`

### `llama_token_fim_suf`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3346`

### `llama_token_fim_mid`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3355`

### `llama_token_fim_pad`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3364`

### `llama_token_fim_rep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3373`

### `llama_token_fim_sep`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3382`

### `llama_vocab_cls`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3393`

### `llama_tokenize`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3432`

### `llama_token_to_piece`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3484`

### `llama_detokenize`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3535`

### `llama_chat_apply_template`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3591`

### `llama_chat_builtin_templates`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3626`

### `llama_sampler_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3710`

### `llama_sampler_name`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3722`

### `llama_sampler_accept`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3732`

### `llama_sampler_apply`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3742`

### `llama_sampler_reset`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3754`

### `llama_sampler_clone`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3764`

### `llama_sampler_free`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3775`

### `llama_sampler_chain_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3788`

### `llama_sampler_chain_add`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3799`

### `llama_sampler_chain_get`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3809`

### `llama_sampler_chain_n`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3821`

### `llama_sampler_chain_remove`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3832`

### `llama_sampler_init_greedy`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3842`

### `llama_sampler_init_dist`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3848`

### `llama_sampler_init_softmax`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3857`

### `llama_sampler_init_top_k`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3865`

### `llama_sampler_init_top_p`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3876`

### `llama_sampler_init_min_p`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3887`

### `llama_sampler_init_typical`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3898`

### `llama_sampler_init_temp`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3905`

### `llama_sampler_init_temp_ext`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3916`

### `llama_sampler_init_xtc`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3929`

### `llama_sampler_init_top_n_sigma`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3942`

### `llama_sampler_init_mirostat`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3958`

### `llama_sampler_init_mirostat_v2`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3974`

### `llama_sampler_init_grammar`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:3990`

### `llama_sampler_init_grammar_lazy`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4018`

### `llama_sampler_init_grammar_lazy_patterns`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4053`

### `llama_sampler_init_penalties`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4077`

### `llama_sampler_init_dry`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4111`

### `llama_sampler_init_logit_bias`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4134`

### `llama_sampler_init_infill`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4147`

### `llama_sampler_get_seed`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4158`

### `llama_sampler_sample`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4169`

### `llama_split_path`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4186`

### `llama_split_prefix`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4205`

### `llama_print_system_info`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4220`

### `llama_log_set`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4232`

### `llama_perf_context`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4287`

### `llama_perf_context_print`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4297`

### `llama_perf_context_reset`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4307`

### `llama_perf_sampler`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4318`

### `llama_perf_sampler_print`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4328`

### `llama_perf_sampler_reset`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4338`

### `llama_opt_param_filter_all`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4357`

### `llama_opt_init`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4386`

### `llama_opt_epoch`
**Location:** `llamarcp-wrapper/llamarcp/llamarcp_wrapper.py:4411`
