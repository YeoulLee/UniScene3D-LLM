"""Parameter-free sinusoidal 3D positional encoding (Video-3D-LLM style).

Produces an embedding of exactly ``dim`` channels from a normalized (x, y, z) coordinate
so it can be added directly to the patch tokens ("just add", no learnable layer). The
channel budget is split evenly across the three axes; each axis gets a standard
sinusoidal/Fourier encoding of its (already normalized to ~[-1, 1]) coordinate value.
"""

import torch


def _axis_embed(values: torch.Tensor, n_channels: int, num_bands: int, max_freq: float) -> torch.Tensor:
    """Fourier-feature encode a scalar field into ``n_channels`` channels.

    Args:
        values: (..., ) coordinate values for one axis, normalized to ~[-1, 1].
        n_channels: number of output channels for this axis (even).
        num_bands: number of frequency bands (n_channels == 2 * num_bands).
        max_freq: highest frequency multiplier.

    Returns:
        (..., n_channels) tensor of [sin; cos] features.
    """
    # Log-spaced bands in [1, max_freq], scaled by pi so normalized coords span [-pi, pi]*f.
    bands = torch.logspace(
        0.0, float(torch.log10(torch.tensor(max_freq))), steps=num_bands,
        device=values.device, dtype=torch.float32,
    )
    angles = values.unsqueeze(-1).float() * bands * torch.pi  # (..., num_bands)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., 2*num_bands)
    return emb[..., :n_channels]


def sinusoidal_pos_embed_3d(coords: torch.Tensor, dim: int, max_freq: float = 64.0) -> torch.Tensor:
    """Build an additive 3D positional encoding of size ``dim``.

    Args:
        coords: (..., 3) normalized coordinates (expected range ~[-1, 1]).
        dim: output channel count, must match the token dimension.
        max_freq: highest Fourier frequency multiplier.

    Returns:
        (..., dim) positional encoding, dtype/device matching ``coords``.
    """
    per_axis = dim // 3
    if per_axis % 2 == 1:
        per_axis -= 1  # each axis needs an even split for [sin; cos]
    num_bands = max(per_axis // 2, 1)

    embeds = []
    for a in range(3):
        embeds.append(_axis_embed(coords[..., a], per_axis, num_bands, max_freq))
    pe = torch.cat(embeds, dim=-1)  # (..., 3*per_axis)

    remainder = dim - pe.shape[-1]
    if remainder > 0:
        pad = torch.zeros(*pe.shape[:-1], remainder, device=pe.device, dtype=pe.dtype)
        pe = torch.cat([pe, pad], dim=-1)
    return pe.to(coords.dtype)
