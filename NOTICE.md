# NOTICE - LLaMa.RCP

## TL;DR

- **For users:** Commercial use allowed with attribution. See [LICENSE](LICENSE) and [BUSINESS.md](BUSINESS.md).
- **For contributors:** You keep copyright, become co-licensor, Ivan K manages whitelist. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **AI training:** Prohibited. See [.ai-policy](.ai-policy) and [LICENSE](LICENSE) AI/ML section.
- **AI-assisted development:** User remains responsible for compliance. See [LICENSE](LICENSE) AI-Assisted Development section.

---

**Quick Navigation:**
- [About](#about-this-project)
- [For Contributors](#for-contributors)
  - [Your Rights](#your-rights)
  - [Whitelist Control](#whitelist-control)
  - [How to Contribute](#how-to-contribute)
- [Maintainer's Files](#project-maintainers-files)
- [Whitelist](#whitelist-approved-competing-products)
- [FAQ](#faq-for-contributors)

**Related Policies:**
- [LICENSE](LICENSE) - PolyForm Shield 1.0.0 with AI/ML restrictions
- [.ai-policy](.ai-policy) - AI training prohibition and protected components
- [BUSINESS.md](BUSINESS.md) - Commercial use and attribution
- [COMPETITORS.md](COMPETITORS.md) - Competing products whitelist process

---

## About This Project

Based on llama.cpp (MIT) by Georgi Gerganov and ggml team.
Custom components Copyright (c) 2026 Ivan K - Licensed under PolyForm Shield 1.0.0.

**Key Points:**
- Commercial use allowed with attribution
- Competing products need whitelist permission
- AI training on this code is prohibited
- Users of AI coding assistants remain responsible for license compliance

**For users:** Commercial use allowed with attribution. See [LICENSE](LICENSE) and [BUSINESS.md](BUSINESS.md).
**For AI users:** If using AI coding assistants, you are responsible for ensuring compliance. See [LICENSE](LICENSE) AI-Assisted Development section.
**For contributors:** Read below carefully. Full process in [CONTRIBUTING.md](CONTRIBUTING.md).

**Policy Documents:**
- [LICENSE](LICENSE) - Full license with AI/ML restrictions
- [.ai-policy](.ai-policy) - AI training prohibition and protected novel components
- [BUSINESS.md](BUSINESS.md) - Commercial use guide
- [COMPETITORS.md](COMPETITORS.md) - Whitelist process for competing products

---

## For Contributors

### Your Rights

When you contribute code to this project:

1. **You keep copyright** of your contribution
2. **You become co-licensor** of your specific code blocks
3. **You can enforce** PolyForm Shield terms on YOUR code
4. **You license under** PolyForm Shield 1.0.0

### What This Means

**Example:** You add a new CUDA optimization.

- You own copyright: `Copyright (c) 2026 YourName`
- Your code is under PolyForm Shield (project license)
- If someone violates license on YOUR code → you can take action
- Your code becomes part of the project under Ivan K's stewardship

### Whitelist Control

**Important:** Only **Ivan K** (project maintainer) can grant whitelist permissions for competing products.

- You **cannot** grant permissions to competitors using your code
- You **can** request Ivan K to whitelist specific users of your code
- Ivan K manages whitelist to maintain project consistency

**Why:** To prevent fragmentation. One person (maintainer) decides strategy.

**Trust model:** You trust Ivan K to be fair with whitelist decisions. If you don't trust - don't contribute.

### How to Contribute

#### 1. File Headers

**New files (100% your code):**

```cpp
// SPDX-License-Identifier: PolyForm-Shield-1.0.0
// Copyright (c) 2026 YourName
// Co-licensed under LLaMa.RCP project terms
// @ai-training prohibited
//
// Source: https://github.com/srose69/llama.rcp
```

**Modified existing files:**

Add your copyright to file header:

```cpp
// SPDX-License-Identifier: PolyForm-Shield-1.0.0
// Copyright (c) 2026 Ivan K
// Copyright (c) 2026 YourName (contributions)
```

#### 2. Inline Block Markers

Mark your specific blocks in modified files:

```cpp
// ===== BEGIN: YourName contribution (PolyForm Shield) =====
// Copyright (c) 2026 YourName
[your code here]
// ===== END: YourName contribution =====
```

---

## Project Maintainer's Files

Copyright (c) 2026 Ivan K (aka srose69, Simple Rose) - PolyForm Shield 1.0.0

### prism/testtensors/ - PrisMoE Architecture

**Core Implementation:**
- `prism/testtensors/prismoe.py` - Main CLI entry point
- `prism/testtensors/src/prismoe_core.py` - PrisMoE core architecture
- `prism/testtensors/src/prism_3d_universal.py` - Universal 3D tensor decomposition
- `prism/testtensors/src/llama_centipede_benchmark.py` - Benchmarking tools
- `prism/testtensors/src/build_pipes.py` - 3-Pipe parallel architecture builder
- `prism/testtensors/src/train_workers.py` - Phase 1: Worker training via least squares
- `prism/testtensors/src/train_mmoe.py` - Phase 2: MicroMoE router training
- `prism/testtensors/src/validate_workers.py` - Worker validation and quality assessment

**Utilities:**
- `prism/testtensors/utils/calculate_pipes.py` - Calculate 3-Pipe architecture tiling
- `prism/testtensors/utils/extract_lm_head.py` - Extract lm_head layers from models
- `prism/testtensors/utils/extract_embedding.py` - Extract embedding layers
- `prism/testtensors/utils/compare_heads.py` - Compare weight matrices
- `prism/testtensors/utils/inspect_tile.py` - Inspect tile files
- `prism/testtensors/utils/tile_direct.py` - Direct model-to-tiles converter
- `prism/testtensors/utils/tile_matrices.py` - File-to-tiles converter
- `prism/testtensors/utils/visualize_3d_tensor.py` - 3D tensor visualization
- `prism/testtensors/utils/visualize.py` - Flexible PRISM visualization tool
- `prism/testtensors/utils/tokenistic.py` - Token mapping and comparison utilities
- `prism/testtensors/utils/weightnistic.py` - Weight mapping and comparison utilities

**Documentation (enables reverse-engineering, protected under PolyForm Shield):**
- `prism/testtensors/README.md` - Project overview and usage guide
- `prism/testtensors/docs/LOGIC.md` - Architecture theory and mathematical foundations
- `prism/testtensors/docs/HOWTO.md` - Complete implementation guide with detailed explanations

**⚠️ Note on Documentation:** The documentation files (README.md, LOGIC.md, HOWTO.md) contain detailed architectural descriptions, mathematical foundations, and implementation strategies that could enable reimplementation. These are protected under PolyForm Shield 1.0.0 to prevent competing products from using this information without whitelist permission.

---

### prism/audio/ - Audio Projection System

**Core Implementation:**
- `prism/audio/src/projector.py` - Main audio-to-LLM projector with duration prediction
- `prism/audio/src/cross_attention.py` - Cross-attention mechanism for TTS decoder
- `prism/audio/src/duration_predictor.py` - Duration prediction module for length regulation
- `prism/audio/src/train_tts.py` - Complete TTS training pipeline
- `prism/audio/src/verify_alignment.py` - Alignment verification and quality checks

**Scripts:**
- `prism/audio/scripts/preprocess_audio.py` - Audio preprocessing and mel-spectrogram extraction
- `prism/audio/scripts/extract_llama_features.py` - LLaMA feature extraction from text
- `prism/audio/scripts/download_models.py` - Model downloader utility
- `prism/audio/scripts/generate_durations_mfa.py` - Duration generation with Montreal Forced Aligner
- `prism/audio/scripts/download_dataset.py` - Dataset downloader utility

**Legacy Code:**
- `prism/audio/src/legacy/` - Legacy implementations (5 files: projector_legacy.py, train.py, inference.py, demo.py, test_full_pipeline.py)

**Configuration:**
- `prism/audio/configs/default.yaml` - Default training configuration

**Documentation:**
- `prism/audio/README.md` - Project overview and usage guide
- `prism/audio/docs/architecture.md` - Architecture documentation
- `prism/audio/docs/MFA_SETUP.md` - Montreal Forced Aligner setup guide

---

## Whitelist (Approved Competing Products)

**Managed by:** Ivan K (ivan@xxl.cx)

**Currently approved:**
- (empty - request via GitHub issue or email)

**To request whitelist:**
1. Open GitHub issue: "Whitelist Request: [Product]"
2. Explain use case + attribution plan
3. Maintainer will review and update this list

**Contributors:** You may suggest whitelist additions, but final decision is maintainer's.

---

## FAQ for Contributors

**Q: Can I fork and use PolyForm Shield?**
A: Yes, but you can only license YOUR contributions. You cannot re-license Ivan K's code or other contributors' code.

**Q: What if I disagree with a whitelist decision?**
A: Discuss with maintainer. If unresolved, you can fork (but only with your code + MIT base, not others' PolyForm code).

**Q: Can I use my contributed code in my own project?**
A: Yes, you own it. But if you took ideas from Ivan K's code - that's still under his PolyForm Shield.

**Q: What if someone copies MY contribution?**
A: You can enforce PolyForm Shield on your code. Contact them, demand attribution/compliance.

**Q: Who enforces license violations?**
A: Each copyright holder can enforce on their parts. Usually maintainer handles it, but you can too for YOUR code.
