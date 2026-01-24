# Contributing to LLaMa.RCP

## TL;DR

- Your code → PolyForm Shield 1.0.0 ([LICENSE](LICENSE))
- You keep copyright, become co-licensor ([NOTICE.md](NOTICE.md#for-contributors))
- Ivan K manages whitelist (you trust him)
- Add headers, sign-off in PR
- See below for details

---

**Quick Navigation:**
- [License Agreement](#license-agreement)
- [Contributor License Agreement](#contributor-license-agreement)
- [Acceptable Contributions](#acceptable-contributions)
- [Process](#process)
- [Code Style](#code-style)
- [Upstream Sync](#upstream-sync)
- [Questions](#questions)

---

## License Agreement

By submitting a pull request:
1. You agree your code will be under PolyForm Shield 1.0.0 ([LICENSE](LICENSE))
2. You grant the project maintainer right to relicense your contribution
3. You retain copyright of your contribution (see [NOTICE.md](NOTICE.md#your-rights)) to the whitelist for commercial use
3. You retain copyright of your work
4. You agree to the AI training prohibition

## Contributor License Agreement

By contributing, you grant Ivan K:
- Right to relicense ONLY for whitelist purposes
- Right to sublicense to approved commercial users
- You keep your copyright
- Your code stays PolyForm Shield by default

## Acceptable Contributions

✅ Bug fixes
✅ Performance improvements
✅ New features (that don't compete with core mission)
✅ Documentation
✅ Tests

❌ Code that would help competitors bypass noncompete clause
❌ Removing attribution requirements
❌ Weakening PolyForm Shield protections

## Process

1. Fork the repo
2. Create feature branch
3. Add proper headers to new files (see NOTICE.md for templates)
4. Submit PR with clear description
5. Sign-off: "I agree to PolyForm Shield 1.0.0 terms"

## Code Style

Follow existing code style:
- C++17/20 standards
- Descriptive variable names
- Minimal but informative comments
- CUDA: PTX assembly where beneficial
- Rust: explicit lifetimes, no unwrap()

## Upstream Sync

We sync with [llama.cpp](https://github.com/ggml-org/llama.cpp) upstream to get improvements and fixes.

When contributing:
- Keep compatibility with upstream code structure where possible
- Report bugs to llama.cpp if they're in base code (and here, it may be fixed faster)
- Contribute fixes upstream when applicable
- Contribute sync with upstream when applicable

This helps both projects improve.

---

## Questions?

Open an issue or contact ivan@xxl.cx
