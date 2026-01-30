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

**Pascal GPU Optimizations (60.12 t/s achievement):**
- `ggml/src/ggml-cuda/device_async_gate.cuh` - Warp-level async soft-gate synchronization primitives
- `ggml/src/ggml-cuda/mmq.cuh` - Q4/Q8 aggressive L1/L2 prefetch, async gate integration
- `ggml/src/ggml-cuda/mmq_q4.cuh` - Zero-Cost SWAR unpacking (dual-bank shared memory layout)
- `ggml/src/ggml-cuda/vecdotq_q4.cuh` - PTX-optimized vec_dot with separate lo/hi bank loads
- `ggml/src/ggml-cuda/mmvq_ptx.cuh` 
- `README.md` - Performance metrics update
- `.gitignore` - Analysis artifacts exclusion


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
