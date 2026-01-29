# Contributing to LLaMa.RCP / RnD Branch

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
- [Questions](#questions)

---

## License Agreement

By submitting a pull request:
1. You agree your code will be under PolyForm Shield 1.0.0 ([LICENSE](LICENSE))
2. You grant the project maintainer right to relicense your contribution for whitelist purposes
3. You retain copyright of your contribution (see [NOTICE.md](NOTICE.md#for-contributors))
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
✅ New research ideas (discuss first)
✅ Documentation improvements
✅ Test coverage
✅ Dataset/training scripts

❌ Code that would help competitors bypass noncompete clause
❌ Removing attribution requirements
❌ Weakening PolyForm Shield protections

## Process

1. **Open an issue first** - Discuss your idea/bug
2. **Fork the repo** - Create your branch
3. **Add proper headers** to new files (see NOTICE.md for templates)
4. **Write tests** - Ensure quality
5. **Submit PR** with clear description

### For Research Contributions

If you're contributing a new research idea:
1. Open an issue describing the approach
2. Discuss mathematical foundations/architecture
3. Get feedback before implementation
4. Submit with detailed documentation

## Code Style

Follow existing code style:

**Python:**
- PEP 8 compliant
- Type hints required (`def func(x: int) -> str:`)
- Docstrings (Google style)
- Descriptive variable names
- Minimal but informative comments

**PyTorch:**
- Use torch.nn.Module for models
- Register buffers/parameters properly
- Document tensor shapes in comments
- Use device-agnostic code (`tensor.to(device)`)

**Audio Processing:**
- Document sample rates and frame rates
- Validate input/output shapes
- Handle edge cases (empty sequences, etc.)

**Example:**
```python
"""
Module for audio feature projection.

This module implements...
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AudioProjector(nn.Module):
    """
    Projects LLM hidden states to TTS feature space.
    
    Args:
        input_dim: LLM hidden dimension (e.g., 4096)
        output_dim: TTS feature dimension (e.g., 2048)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Implementation...
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, input_dim]
    ) -> torch.Tensor:  # [batch, seq_len, output_dim]
        """Project hidden states to TTS features."""
        # Implementation...
```

---

## Questions?

Open an issue or contact ivan@xxl.cx

---

© 2026 Ivan K - All Rights Reserved (PolyForm Shield 1.0.0)
