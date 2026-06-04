"""Parameter-free sinusoidal 3D positional encoding (Video-3D-LLM, arXiv:2412.00493).

Matches the paper's formulation: each axis (x, y, z) is encoded with the standard
transformer sinusoidal PE applied to the *raw* coordinate, the channel budget split as
floor(d/3) per axis, and the three encodings concatenated:

    PE(x, 2i)   = sin( x / 10000^(2i / d_axis) )
    PE(x, 2i+1) = cos( x / 10000^(2i / d_axis) )

The result is added directly to the visual tokens (no learnable layer). The paper feeds
raw metric coordinates (no per-scene normalization); coordinate scaling is handled upstream
in frame.apply_coord_frame via the configurable `normalize` option (default "none").
"""

import torch


def _axis_embed(values: torch.Tensor, n_channels: int, temperature: float) -> torch.Tensor:
    """Transformer sinusoidal PE of a scalar coordinate field into ``n_channels`` channels.

    Args:
        values: (...,) raw coordinate values for one axis.
        n_channels: even number of output channels for this axis.
        temperature: base of the geometric frequency progression (paper uses 10000).

    Returns:
        (..., n_channels) tensor with interleaved [sin, cos] per frequency.
    """
    half = n_channels // 2
    i = torch.arange(half, device=values.device, dtype=torch.float32)
    div_term = temperature ** (-(2.0 * i) / n_channels)   # 1 / 10000^(2i/d_axis)
    ang = values.unsqueeze(-1).float() * div_term         # (..., half)

    emb = torch.empty(*ang.shape[:-1], n_channels, device=values.device, dtype=torch.float32)
    emb[..., 0::2] = torch.sin(ang)   # even indices -> sin
    emb[..., 1::2] = torch.cos(ang)   # odd indices  -> cos
    return emb


def sinusoidal_pos_embed_3d(coords: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """Build an additive 3D positional encoding of size ``dim`` (Video-3D-LLM style).

    Args:
        coords: (..., 3) coordinates (raw metric, unless normalized upstream).
        dim: output channel count, must match the token dimension.
        temperature: sinusoidal base (paper: 10000).

    Returns:
        (..., dim) positional encoding, dtype/device matching ``coords``.
    """
    per_axis = dim // 3
    if per_axis % 2 == 1:
        per_axis -= 1  # each axis needs an even split for [sin, cos]

    embeds = [_axis_embed(coords[..., a], per_axis, temperature) for a in range(3)]
    pe = torch.cat(embeds, dim=-1)  # (..., 3*per_axis)

    remainder = dim - pe.shape[-1]
    if remainder > 0:
        pad = torch.zeros(*pe.shape[:-1], remainder, device=pe.device, dtype=pe.dtype)
        pe = torch.cat([pe, pad], dim=-1)
    return pe.to(coords.dtype)
