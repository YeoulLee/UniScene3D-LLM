"""Token-reduction stage (pluggable).

The reducer compresses the flattened visual token set before it enters the projector/LLM.
For now only ``Identity`` (no reduction) is used, matching the "simplest version" plan;
future plug-ins (voxel pooling, pixel-shuffle, FPS, Q-Former) register here and are selected
by ``cfg.model.reducer.name`` with no change to the model code.

Reducer contract (all operate on the flattened token set):
    forward(tokens, coords, valid) -> (tokens', coords', valid')
        tokens: (B, N, D)
        coords: (B, N, 3)   normalized coordinates aligned with tokens
        valid:  (B, N)      bool mask of usable tokens
"""

import torch.nn as nn
from fvcore.common.registry import Registry

from common.misc import cfg2dict

TOKEN_REDUCER_REGISTRY = Registry("token_reducer")


@TOKEN_REDUCER_REGISTRY.register()
class Identity(nn.Module):
    """Pass-through reducer: keep every token (N = V * P)."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, tokens, coords, valid):
        return tokens, coords, valid


def build_token_reducer(cfg_reducer):
    """Build a reducer from ``cfg.model.reducer`` (expects ``.name`` and optional ``.args``)."""
    name = cfg_reducer.get("name", "Identity")
    args = cfg2dict(cfg_reducer.get("args", {})) if cfg_reducer.get("args", None) else {}
    return TOKEN_REDUCER_REGISTRY.get(name)(**args)
