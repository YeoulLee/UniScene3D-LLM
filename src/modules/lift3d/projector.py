"""Vision-to-LLM projector (LLaVA-style MLP).

Maps each visual token from the FG-CLIP feature dimension (512 projected / 768 penultimate)
to the LLM hidden size before the tokens are spliced into the Qwen input sequence.
"""

import torch.nn as nn


class Projector(nn.Module):
    """Two-layer GELU MLP projecting visual tokens into the LLM embedding space."""

    def __init__(self, in_dim: int, llm_hidden: int, hidden_dim: int = None, depth: int = 2):
        """Build the projector.

        Args:
            in_dim: input feature dimension (512 or 768).
            llm_hidden: target LLM hidden size.
            hidden_dim: MLP hidden width (defaults to llm_hidden).
            depth: number of linear layers (>=1). depth==1 is a single Linear.
        """
        super().__init__()
        hidden_dim = hidden_dim or llm_hidden
        if depth <= 1:
            self.mlp = nn.Linear(in_dim, llm_hidden)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(depth - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, llm_hidden)]
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
