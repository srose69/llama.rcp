# For Competitors - LLaMa.RCP / RnD Branch

## TL;DR

**Want to compete?** Get whitelist permission first.
**Better idea?** Collaborate instead - pool resources, share success.
**Whitelist is usually FREE.** Just ask nicely.
**For regular commercial use:** See [LICENSE](LICENSE)
**Collaboration = automatic approval + shared credit.**

---

**Quick Navigation:**
- [TL;DR](#tldr)
- [Scope of Competition](#scope-of-competition)
- [Collaborate > Compete](#why-collaborate-instead-of-compete)
- [What Counts as Competition](#what-counts-as-competition)
- [Whitelist Process](#whitelist-process)
- [Contact](#contact)

---

## Scope of Competition

**IMPORTANT:** This policy applies to **research in this specific RnD branch**, not to the entire TTS/AI field.

### What We Do (and compete in):
- **LLM-to-TTS projection adapters** - Bridging LLM hidden states to TTS feature spaces
- **Multi-modal alignment research** - Novel approaches to feature space mapping
- **Training strategies** for frozen model adapters
- **Specific research implementations** in this repository

### What We DON'T Compete With:
- ❌ **TTS systems** (VITS, FastSpeech, Tacotron, Bark, etc.) - We USE them as decoders
- ❌ **LLM frameworks** (any pretrained LLMs) - We BUILD ON them as encoders
- ❌ Audio processing tools (librosa, MFA, etc.) - These are our DEPENDENCIES
- ❌ Other AI research areas outside our RnD scope

**Think of it this way:**
- If you're building a better TTS decoder → **NOT** competing with us
- If you're building LLM-to-TTS projection → **Potentially** competing (let's collaborate!)
- If you're using our projector in your product → **NOT** competing (just credit us)

---

## Why Collaborate Instead of Compete?

**Think about it:** Instead of building a separate LLM-to-TTS adapter, why not become a **contributor** or **collaborator**?

### About the Maintainer

I'm (Simple Rose / aka srose69 on GitHub) a reasonable person who values good ideas. **Whitelist access is usually FREE** - you don't need to pay anything in most cases. Just ask nicely and show good faith.

**But here's the question:** Why even need a whitelist when you can **build together**?

### Many Bicycles < One Good Car

Fragmentation hurts everyone:
- ❌ 10 separate projection implementations → each with bugs, maintenance burden, limited features
- ✅ 1 solid project with 10 contributors → better quality, shared workload, faster progress

**Benefits of collaboration:**

- ✅ **Pool resources** - Don't duplicate research effort
- ✅ **Better product** - More contributors = more features, fewer bugs
- ✅ **Shared maintenance** - You're not alone fixing issues
- ✅ **Influence roadmap** - Shape the project direction
- ✅ **Credit & recognition** - Co-author status, your name in commits/papers
- ✅ **Automatic whitelist** - Contributors get approved by default

### If Our Goals Align

- You have better duration predictor → Contribute it → Everyone benefits + you get whitelist
- You found optimization for projection architecture → PR it → Both use it
- You need different decoder support → Let's discuss → Maybe collaborate on implementation

**Good ideas deserve to be shared, not locked in competing forks.**

### This RnD Branch Has Many Projects

PRISM is **ONE** of several research projects here. More coming:
- Future multi-modal experiments
- Different alignment approaches
- Novel training strategies

**Collaborating on one project doesn't lock you into others.** Each project stands alone.

---

## What Counts as Competition?

### ❌ Competing Products (need whitelist):

**LLM-to-TTS projection systems:**
- Fork this to create "MyProjection Framework" for LLM→TTS
- Build a product marketed as "faster alternative to PRISM"
- Offer "LLM-to-TTS projection as a service" using our code
- Create closed-source projector using our architecture

**Extract research to compete:**
- Take TTSProjector architecture for commercial projection product
- Use our duration prediction approach in competing projector
- Extract our alignment strategies for competing implementation
- Reproduce novel components (see NOTICE.md for protected research)

### ✅ NOT Competition (no whitelist needed):

**Using our projector as a component:**
- Voice assistant using our TTS pipeline
- Audiobook generation platform  
- Accessibility tools (text-to-speech for blind users)
- Educational content creation
- **ANY product where LLM-to-TTS projection is a FEATURE, not the PRODUCT**

**Building different projection types:**
- Image-to-Text projection (different input modality)
- LLM-to-Music generation (audio, but not TTS)
- Text-to-3D (different output space)
- Any other modality combinations (Video, 3D, etc.)

**Using different approaches:**
- Direct TTS without LLM projection (e.g., standalone TTS models)
- LLM with voice cloning (different architecture)
- Traditional TTS pipelines

**Research papers:**
- Citing algorithms/approaches in academic work
- Implementing ideas in different research context
- Comparative studies

**Think of it this way:**
- If you're selling "better LLM-to-TTS projection" → need whitelist
- If you're selling "better audiobooks" (using our projection) → just credit us
- If you're researching "Image-to-Audio" → totally different, no whitelist needed

---

## Whitelist Process

### How to Get Whitelist

1. **Open GitHub issue:** "Whitelist Request: [Your Product]"
2. **Describe:**
   - What you're building
   - How it competes with THIS specific research (be honest)
   - How you'll attribute
   - Why you can't collaborate instead
3. **I'll review** and update [NOTICE.md](NOTICE.md#whitelist-approved-competing-products) if approved

**Or email:** ivan@xxl.cx

### Decision Criteria

I'll likely approve if:
- ✅ Legitimate use case
- ✅ Good faith request
- ✅ Willing to attribute properly
- ✅ Respectful approach

I'll likely deny if:
- ❌ Trying to hide origins
- ❌ Planning to claim originality
- ❌ Bad faith / disrespectful

### Whitelist is FREE

**You don't pay money.** Just:
- Be honest about your plans
- Give proper attribution
- Show good faith

---

## Contact

**Collaboration discussion:** ivan@xxl.cx
**Whitelist request:** GitHub issue or ivan@xxl.cx

**Prefer collaboration?** Let's talk. Building together > building separately.

---

© 2026 Ivan K - All Rights Reserved (PolyForm Shield 1.0.0)
