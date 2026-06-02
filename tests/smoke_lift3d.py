"""CPU smoke test for the 3D-lifting math (no transformers / no GPU required).

Run on the cluster (or any box with torch):  python tests/smoke_lift3d.py

Validates coord_pool (incl. invalid-pixel exclusion), the world/ego frame transform,
per-scene normalization range, the additive 3D PE shape, the Identity reducer, and the
projector shape. This covers Phase 2 of the build order before any LLM/data is wired.
"""

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from modules.lift3d import (  # noqa: E402
    pool_pointmap_to_patches,
    apply_coord_frame,
    normalize_coords,
    sinusoidal_pos_embed_3d,
    build_token_reducer,
    Projector,
)
from modules.lift3d.frame import transform_to_ego  # noqa: E402


def test_coord_pool_invalid_exclusion():
    B, V, H, W, P = 2, 3, 32, 32, 16  # grid 4x4, kernel 8x8
    pm = torch.zeros(B, V, 3, H, W)
    # Fill the top-left 8x8 patch (patch index 0) with a constant valid coordinate.
    pm[0, 0, :, 0:8, 0:8] = torch.tensor([2.0, 3.0, 1.0]).view(3, 1, 1)
    # Add a single noisy "hole-like" zero pixel inside that block; it must be excluded.
    pm[0, 0, :, 0, 0] = 0.0
    coords, valid = pool_pointmap_to_patches(pm, P)
    assert coords.shape == (B, V, P, 3), coords.shape
    assert valid.shape == (B, V, P)
    # Patch 0 should equal the constant (zeros excluded), not dragged toward origin.
    assert torch.allclose(coords[0, 0, 0], torch.tensor([2.0, 3.0, 1.0]), atol=1e-4), coords[0, 0, 0]
    assert bool(valid[0, 0, 0])
    # A fully-zero patch is invalid and its coordinate stays ~0.
    assert not bool(valid[0, 0, 5])
    assert torch.allclose(coords[0, 0, 5], torch.zeros(3))
    print("[ok] coord_pool excludes invalid pixels and flags empty patches")


def test_ego_puts_anchor_at_origin():
    B, V, P = 2, 1, 4
    coords = torch.randn(B, V, P, 3)
    anchor_loc = torch.tensor([[1.0, 2.0, 0.5], [-1.0, 0.0, 3.0]])
    anchor_yaw = torch.tensor([0.7, -1.2])
    # Put the anchor location itself as one of the points; it must map to the origin.
    coords[:, 0, 0] = anchor_loc
    ego = transform_to_ego(coords, anchor_loc, anchor_yaw)
    assert torch.allclose(ego[:, 0, 0], torch.zeros(B, 3), atol=1e-5), ego[:, 0, 0]
    print("[ok] ego transform maps the agent anchor to the origin")


def test_normalize_range():
    B, V, P = 2, 4, 16
    coords = torch.randn(B, V, P, 3) * 5.0
    valid = torch.ones(B, V, P, dtype=torch.bool)
    valid[0, 0, :] = False  # some invalid patches must not break stats
    normed = normalize_coords(coords, valid)
    finite_valid = normed[valid]
    assert finite_valid.abs().max() <= 1.0 + 1e-4, finite_valid.abs().max()
    print("[ok] per-scene normalization keeps valid coords within [-1, 1]")


def test_frame_switch():
    B, V, P = 2, 2, 16
    coords = torch.randn(B, V, P, 3)
    valid = torch.ones(B, V, P, dtype=torch.bool)
    w = apply_coord_frame(coords, valid, "world")
    e = apply_coord_frame(coords, valid, "ego",
                          anchor_loc=torch.randn(B, 3), anchor_yaw=torch.randn(B))
    assert w.shape == e.shape == (B, V, P, 3)
    assert not torch.allclose(w, e), "world and ego frames should differ"
    print("[ok] world/ego frame switch produces distinct normalized coords")


def test_pos_embed_and_projector():
    B, V, P, D, Hllm = 2, 3, 16, 512, 64
    coords = torch.rand(B, V, P, 3) * 2 - 1
    pe = sinusoidal_pos_embed_3d(coords, D)
    assert pe.shape == (B, V, P, D), pe.shape
    assert torch.isfinite(pe).all()

    tokens = torch.randn(B, V * P, D) + pe.reshape(B, V * P, D)
    coords_f = coords.reshape(B, V * P, 3)
    valid_f = torch.ones(B, V * P, dtype=torch.bool)
    reducer = build_token_reducer({"name": "Identity"})
    t2, c2, v2 = reducer(tokens, coords_f, valid_f)
    assert t2.shape == tokens.shape and c2.shape == coords_f.shape

    proj = Projector(in_dim=D, llm_hidden=Hllm, depth=2)
    out = proj(t2)
    assert out.shape == (B, V * P, Hllm), out.shape
    print("[ok] 3D PE adds cleanly, Identity reducer passes through, projector maps to LLM dim")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_coord_pool_invalid_exclusion()
    test_ego_puts_anchor_at_origin()
    test_normalize_range()
    test_frame_switch()
    test_pos_embed_and_projector()
    print("\nAll lift3d smoke tests passed.")
