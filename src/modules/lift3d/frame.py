"""Coordinate-frame handling for the global 3D positional encoding.

Two frames, selected by ``cfg.model.coord_frame``:

* ``world`` : keep raw world coordinates. Only per-scene normalization is applied.
* ``ego``   : rotate + translate every coordinate into the agent's situated frame using
              the SQA3D anchor pose, so "left/right/front/behind" map to coordinate-sign
              structure that the 3D PE can express directly. Then normalize.

Only the ego branch differs in code; normalization is shared by both frames.

Convention for the ego frame: the agent is placed at the origin. We rotate about the
vertical (z) axis by ``-yaw`` so the agent's heading aligns with +y (forward); +x is then
to the agent's right. (z stays the world up-axis. SQA3D agents are floor-bound, so a single
yaw captures the heading.)
"""

import torch


def transform_to_ego(coords: torch.Tensor, anchor_loc: torch.Tensor, anchor_yaw: torch.Tensor) -> torch.Tensor:
    """Map world coordinates into the agent-centric (ego) frame.

    Args:
        coords: (B, V, P, 3) world coordinates.
        anchor_loc: (B, 3) agent position in world frame.
        anchor_yaw: (B,) agent heading in radians (rotation about +z).

    Returns:
        (B, V, P, 3) coordinates in the ego frame.
    """
    B = coords.shape[0]
    # Translate so the agent sits at the origin.
    centered = coords - anchor_loc.view(B, 1, 1, 3)

    cos = torch.cos(-anchor_yaw)  # rotate by -yaw to undo the agent's heading
    sin = torch.sin(-anchor_yaw)
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    # Rotation about z: rows are [x', y', z'].
    rot = torch.stack([
        torch.stack([cos, -sin, zero], dim=-1),
        torch.stack([sin, cos, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)  # (B, 3, 3)

    # (B,V,P,3) x (B,3,3)^T  ->  rotate each point.
    ego = torch.einsum("bvpc,bdc->bvpd", centered, rot)
    return ego


def normalize_coords(coords: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-scene min-max normalization of coordinates into [-1, 1].

    Statistics are computed over valid patches only (per batch element, across all views
    and patches) so depth-hole patches do not corrupt the bounding box.

    Args:
        coords: (B, V, P, 3) coordinates.
        valid:  (B, V, P) bool mask of usable patches.
        eps: floor on the per-axis extent to avoid divide-by-zero on degenerate scenes.

    Returns:
        (B, V, P, 3) coordinates normalized to roughly [-1, 1]; invalid patches stay 0.
    """
    B, V, P, _ = coords.shape
    flat = coords.reshape(B, V * P, 3)
    m = valid.reshape(B, V * P, 1).to(coords.dtype)

    very_large = torch.finfo(coords.dtype).max
    masked_min = torch.where(m.bool(), flat, torch.full_like(flat, very_large))
    masked_max = torch.where(m.bool(), flat, torch.full_like(flat, -very_large))
    cmin = masked_min.min(dim=1, keepdim=True).values  # (B,1,3)
    cmax = masked_max.max(dim=1, keepdim=True).values  # (B,1,3)

    # Scenes with no valid patch (should not happen) collapse to a no-op range.
    no_valid = (valid.reshape(B, V * P).sum(dim=1) == 0).view(B, 1, 1)
    cmin = torch.where(no_valid, torch.zeros_like(cmin), cmin)
    cmax = torch.where(no_valid, torch.ones_like(cmax), cmax)

    center = (cmax + cmin) / 2.0
    half_extent = (cmax - cmin).clamp_min(eps) / 2.0
    normed = (flat - center) / half_extent  # -> [-1, 1]
    normed = normed * m  # keep invalid patches at 0
    return normed.reshape(B, V, P, 3)


def apply_coord_frame(coords, valid, frame, normalize="none",
                      anchor_loc=None, anchor_yaw=None, fixed_scale=10.0):
    """Apply the configured coordinate frame, then optional coordinate scaling.

    Video-3D-LLM (arXiv:2412.00493) feeds *raw* metric coordinates to the sinusoidal 3D PE,
    so the default is ``normalize="none"``. Per-scene min-max ("scene_bbox") and fixed metric
    scaling ("fixed_scale") are kept as ablations.

    Args:
        coords: (B, V, P, 3) world coordinates from coord_pool.
        valid:  (B, V, P) bool patch-validity mask.
        frame: "world" or "ego".
        normalize: "none" (paper) | "scene_bbox" | "fixed_scale".
        anchor_loc: (B, 3) required when frame == "ego".
        anchor_yaw: (B,) required when frame == "ego".
        fixed_scale: divisor (meters) used when normalize == "fixed_scale".

    Returns:
        (B, V, P, 3) coordinates ready for the 3D PE.
    """
    if frame == "ego":
        if anchor_loc is None or anchor_yaw is None:
            raise ValueError("coord_frame='ego' requires anchor_loc and anchor_yaw.")
        coords = transform_to_ego(coords, anchor_loc, anchor_yaw)
    elif frame != "world":
        raise ValueError(f"Unknown coord_frame '{frame}' (expected 'world' or 'ego').")

    if normalize == "none":
        return coords
    if normalize == "scene_bbox":
        return normalize_coords(coords, valid)
    if normalize == "fixed_scale":
        return (coords / float(fixed_scale)) * valid.unsqueeze(-1).to(coords.dtype)
    raise ValueError(f"Unknown normalize '{normalize}' (expected none|scene_bbox|fixed_scale).")
