# For Competitors - LLaMa.RCP

## TL;DR

**Want to compete?** Get whitelist permission first.
**Better idea?** Collaborate instead - pool resources, share success.
**Whitelist is usually FREE.** Just ask nicely.
**For regular commercial use:** See [BUSINESS.md](BUSINESS.md)
**Collaboration = automatic approval + shared credit.**

---

**Quick Navigation:**
- [TL;DR](#tldr)
- [Collaborate > Compete](#why-collaborate-instead-of-compete)
  - [Example: llama.cpp](#example-our-relationship-with-llamacpp)
- [What Counts as Competition](#what-counts-as-competition)
- [Whitelist Process](#whitelist-process)
- [Contact](#contact)

---

## Why Collaborate Instead of Compete?

**Think about it:** Instead of building a competing inference engine, why not become a **contributor** or **collaborator**?

### About the Maintainer

 I'm (Simple Rose / aka srose69 on GitHub) a reasonable person who values good ideas. **Whitelist access is usually FREE** - you don't need to pay anything in most cases. Just ask nicely and show good faith.

**But here's the question:** Why even need a whitelist when you can **build together**?

### Many Bicycles < One Good Car

Fragmentation hurts everyone:
- ❌ 10 separate inference engines → each with bugs, maintenance burden, limited features
- ✅ 1 solid project with 10 contributors → better quality, shared workload, faster progress

**Benefits of collaboration:**

- ✅ **Pool resources** - Don't duplicate effort
- ✅ **Better product** - More contributors = more features, fewer bugs
- ✅ **Shared maintenance** - You're not alone fixing issues
- ✅ **Influence roadmap** - Shape the project direction
- ✅ **Credit & recognition** - Co-author status, your name in commits
- ✅ **Automatic whitelist** - Contributors get approved by default

### If Our Goals Align

- You want feature X → Contribute it → Everyone benefits + you get whitelist
- You have optimization Y → PR it → Both use it
- You need capability Z → Let's discuss → Maybe collaborate on implementation

**Good ideas deserve to be shared, not locked in competing forks.**

**Remember:** "Competitor" = separate paths, duplicated work, slower progress. "Collaborator" = shared success, better product, faster iterations. If ideas don't conflict - **why compete when you can build together?**

### Example: Relationship with llama.cpp

LLaMa.RCP builds on [llama.cpp](https://github.com/ggml-org/llama.cpp), not as a competitor but as a specialized variant. See [CONTRIBUTING.md](CONTRIBUTING.md#upstream-sync) for sync process.

- We sync improvements from upstream
- We contribute fixes back when applicable  
- We recommend llama.cpp for general use
- We focus on x86_64 + GPU optimization

**This is collaboration, not competition.** Both projects benefit.

---

## What Counts as Competition?

### ❌ Competing Products (need whitelist):

**LLM inference engines:**
- Fork this to create "MyLLM Inference Framework"
- Build a product marketed as "faster alternative to LLaMa.RCP"
- Offer "LLM inference as a service" using this code

**Extract algorithms to compete:**
- Take LFP8 quantization to build "LFP8 Inference Engine"
- Use Max-Plus Algebra implementation in "MaxPlus Runtime"
- Use Async Softgating in closed-source competing product
- Other proprietary algorithms (see NOTICE.md)

### ✅ NOT Competition (no whitelist needed):

**Use as component in non-competing products:**
- AI chatbot for customer service
- Document analysis tool
- Code generation IDE plugin
- Content creation platform
- Any product where LLM inference is a FEATURE, not the PRODUCT

**Think of it this way:**
- If you're selling "better inference" → need whitelist
- If you're selling "better chatbots using inference" → just credit me

---

## Whitelist Process

### How to Get Whitelist

1. **Open GitHub issue:** "Whitelist Request: [Your Product]"
2. **Describe:**
   - What you're building
   - How it competes (be honest)
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
