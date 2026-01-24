# Commercial Use Guide - LLaMa.RCP

## TL;DR

**✅ Use commercially** - Build products, charge money, keep code private
**✅ Attribution required** - Credit the project in docs/about
**✅ White-label possible** - Request waiver to hide public attribution
**❌ Competing products** - Need permission (see [COMPETITORS.md](COMPETITORS.md))

---

**Quick Navigation:**
- [TL;DR](#tldr)
- [Why Commercial Use Allowed](#why-this-license-allows-commercial-use)
- [Public Attribution Waiver](#how-to-get-public-attribution-waiver)
- [Use Case Examples](#examples---can-i-use-this)
- [Summary Table](#summary-table)
- [Questions](#questions)

**Want to compete?** See [COMPETITORS.md](COMPETITORS.md)

---

## Why This License Allows Commercial Use

### The Core Principle

PolyForm Shield 1.0.0 ([LICENSE](LICENSE)) is **NOT** a "non-commercial" license. It's a **"non-compete"** license.

**What this means:**

- ✅ **Use it in your product** - Build SaaS, enterprise software, closed-source tools
- ✅ **Sell products using it** - Charge money, integrate into paid offerings
- ✅ **Modify and extend** - Add features, optimize, customize
- ✅ **Keep your code private** - No copyleft requirement

**But you MUST:**
- ✅ Give credit (attribution)
- ✅ Not compete directly with LLaMa.RCP itself

---
### About the Maintainer
 I'm (Simple Rose / aka srose69 on GitHub) a reasonable person who values good ideas. **Whitelist access is usually FREE** - you don't need to pay anything in most cases. Just ask nicely and show good faith.
 
## What "Non-Compete" Means

### ❌ You CANNOT do (without permission):

**Build competing LLM inference:**
- "MyLLM Inference Framework" using this code
- Product marketed as "alternative to LLaMa.RCP"
- "LLM inference as a service" based on this

**Extract algorithms to compete:**
- LFP8 quantization → "LFP8 Engine"
- Max-Plus Algebra implementation → "MaxPlus Runtime"
- Async Softgating → closed competing product

### ✅ You CAN do (with attribution):

**Use as component (inference = FEATURE, not PRODUCT):**
- AI chatbot for customer service
- Document analysis tool
- Code generation IDE plugin
- Content creation platform

**Rule of thumb:**
- Selling "better inference" → need permission
- Selling "better chatbots using inference" → just credit

**Want to compete?** See [COMPETITORS.md](COMPETITORS.md) for whitelist process.

---

## How to Get Public Attribution Waiver

### What This Means

By default, you must credit the project **publicly** (README, About page, docs (and if you want, you can hide it from users)).

You can request to **hide public attribution** while keeping code-level attribution.

### Why Would I Grant This?

1. **White-label products** - Your clients don't need to see "Powered by LLaMa.RCP"
2. **Enterprise deployments** - Internal tools don't need public credits
3. **Embedded systems** - No user-facing UI to show attribution

### What You Must Keep (Even With Waiver)

**Always required:**
- Attribution in source code comments
- Attribution in derivative works
- License files (LICENSE, NOTICE.md)
- Copyright notices

### How to Request

**Email:** ivan@xxl.cx

**Include:**
1. **Company/Project name**
2. **Description of product**
3. **How you're using LLaMa.RCP**
4. **Why public attribution is problematic**
5. **Commitment to maintain code-level attribution**

**Example:**
```
Subject: Public Attribution Waiver Request - Super-Mega-LLM Corp

Hi Ivan,

We're Super-Mega-LLM Corp building an enterprise document analysis platform.
We use LLaMa.RCP for on-premise LLM inference.

Our enterprise clients require white-label deployment without 
third-party branding in the UI.

We commit to:
- Maintain full attribution in source code
- Include [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) in distributions
- Not compete with LLaMa.RCP and llama.cpp
- Recommend LLaMa.RCP and llama.cpp when asked about inference engine

Can we get a public attribution waiver?

Thanks,
Super-Mega-LLM Corp
```

### Decision Criteria

I'll likely grant if:
- ✅ Legitimate white-label / enterprise use case
- ✅ Not competing with LLaMa.RCP
- ✅ Willing to keep code attribution
- ✅ Respectful request

I'll likely deny if:
- ❌ Trying to hide origins to compete
- ❌ Planning to claim originality
- ❌ Building a direct competitor

---

## Whitelist System

### Get Pre-Approved for Competing Products

If you want to build something that DOES compete, you can still get permission.

**Process:**

1. Open GitHub issue: "Whitelist Request: [Your Product]"
2. Describe your product honestly
3. Explain how you'll attribute
4. I'll add you to NOTICE.md whitelist

**Example Approved Projects:**

```
✅ FastInfer - High-speed LLM inference service
   Uses: LFP8 quantization
   Attribution: "LFP8 quantization by [Your Name]"
   Granted: 2026-01-24
```

**Why I Grant These:**

- Good faith actors who ask 
- Clear attribution plan
- Contributes back (PRs, bug reports)
- Helps ecosystem grow

---

## Examples - Can I Use This?

### ✅ YES (with public attribution):

**Q: Enterprise SaaS - Document processing platform with AI analysis**
A: Yes. LLM inference is a feature. Add "Powered by LLaMa.RCP" in About/docs.

**Q: Corporate internal tools - Employee knowledge base with LLM search**
A: Yes. Internal use still needs attribution in docs (employees may see it).

**Q: B2B software - Customer service automation platform**
A: Yes. Chatbots use inference as a component. Credit in product docs.

**Q: Financial services - Automated report generation with LLM**
A: Yes. Inference powers features, not the product itself. Attribution required.

**Q: Healthcare - Medical records analysis assistant**
A: Yes. LLM processes data as part of larger system. Credit in documentation.

**Q: E-commerce - Product recommendation engine with LLM**
A: Yes. Recommendations are the product, inference is the tool. Attribute properly.

### ✅ YES (with waiver):

**Q: White-label SaaS - Reselling to enterprise clients under their brand**
A: Yes with waiver. Clients don't need to see "Powered by LLaMa.RCP" in UI directly. Keep hidden & code attribution.

**Q: OEM hardware - Embedded LLM in IoT devices**
A: Yes with waiver. No user-facing UI anyway. Maintain hidden & code attribution.

**Q: Government contracts - Classified/secure deployments**
A: Yes with waiver. Can't publicly disclose tech stack. Keep hidden & code attribution.

### ❌ NO (need whitelist):

**Q: Inference-as-a-Service - API service offering LLM inference**
A: No without whitelist. Inference IS your product → competing.

**Q: Optimized runtime - "FastLLM" using our quantization techniques**
A: No. Extracting algorithms to compete → need permission.

**Q: Managed inference platform - "LLM Cloud" hosting service**
A: No. Direct competition with inference engines → whitelist required.

### ⚠️ MAYBE (contact us):

**Q: AI Platform - Inference + training + fine-tuning + deployment**
A: Probably yes. Inference is one feature among many. Contact to confirm.

**Q: Developer tools - IDE with built-in LLM code completion**
A: Probably yes. Code completion is the product, inference is the engine. Verify.

**Q: Consulting services - Deploying LLM solutions for clients**
A: Yes. You're providing services, not selling competing engine. Attribution in deliverables.

---

## Summary Table

| Use Case | Public Attribution | Code Attribution | Permission Needed |
|----------|-------------------|------------------|-------------------|
| Use in commercial product | Required | Required | No |
| Sell products using this | Required | Required | No |
| White-label deployment | Waivable | Required | Waiver request |
| Embedded systems | Not applicable | Required | Waiver if needed |
| Competing product | Required | Required | Whitelist |
| Extract algorithms to compete | Required | Required | Whitelist |

---

## Questions?

**General use:** Read this doc carefully first.
**Waiver request:** Email [your email]
**Whitelist request:** Open GitHub issue
**Clarification:** Open GitHub discussion

---

**Remember:** YOU CAN use this commercially. I just want credit and to prevent direct competition without collaboration. Be honest, give credit, and we're good. 👍
