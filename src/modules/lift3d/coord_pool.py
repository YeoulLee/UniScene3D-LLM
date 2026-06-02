"""Pool a dense point map into one 3D coordinate per ViT patch token.

The FG-CLIP encoder emits ``P`` patch tokens per view (P = grid*grid, e.g. 196 for a
14x14 grid over a 224px image with patch size 16). Each patch covers a (H/grid, W/grid)
pixel block of the point map, where every pixel already carries a world-frame (x, y, z).

We reduce each block to a single coordinate by averaging ONLY over valid pixels.
Invalid pixels (depth holes -> all-zero or non-finite point-map entries) would otherwise
drag the patch coordinate toward the origin, so they are excluded from the mean. A patch
whose block has no valid pixel is marked invalid in the returned mask.
"""

import math

import torch
import torch.nn.functional as F


def _pixel_valid_mask(point_map: torch.Tensor) -> torch.Tensor:
    """Return a per-pixel validity mask for a point map.

    Args:
        point_map: (BV, 3, H, W) world coordinates.

    Returns:
        (BV, 1, H, W) float mask, 1.0 where the pixel has a usable coordinate.
    """
    finite = torch.isfinite(point_map).all(dim=1, keepdim=True)
    nonzero = point_map.abs().sum(dim=1, keepdim=True) > 0
    return (finite & nonzero).to(point_map.dtype)


def pool_pointmap_to_patches(point_map: torch.Tensor, num_patches: int):
    """Average a point map down to one coordinate per patch token.

    Args:
        point_map: (B, V, 3, H, W) world-frame coordinates per pixel.
        num_patches: number of patch tokens per view (must be a perfect square,
            e.g. 196 -> 14x14 grid).

    Returns:
        coords: (B, V, num_patches, 3) per-patch coordinate (0 where invalid).
        valid:  (B, V, num_patches) bool mask, True where the patch has >=1 valid pixel.
    """
    B, V, C, H, W = point_map.shape
    assert C == 3, f"point_map must have 3 channels, got {C}"

    grid = int(round(math.sqrt(num_patches)))
    assert grid * grid == num_patches, (
        f"num_patches={num_patches} is not a perfect square; cannot map to a patch grid."
    )
    assert H % grid == 0 and W % grid == 0, (
        f"point map size ({H}x{W}) is not divisible by the patch grid ({grid}x{grid}). "
        "Pool adaptively or resize the point map to match the ViT grid."
    )
    kh, kw = H // grid, W // grid

    pm = point_map.reshape(B * V, C, H, W).float()
    valid = _pixel_valid_mask(pm)  # (BV, 1, H, W)

    # avg_pool2d gives the block mean; multiply by block area to recover the block sum.
    block_area = kh * kw
    coord_sum = F.avg_pool2d(pm * valid, kernel_size=(kh, kw)) * block_area  # (BV,3,grid,grid)
    valid_count = F.avg_pool2d(valid, kernel_size=(kh, kw)) * block_area     # (BV,1,grid,grid)

    coords = coord_sum / valid_count.clamp_min(1.0)  # safe where count==0 (coords stay ~0)
    patch_valid = valid_count.squeeze(1) > 0          # (BV, grid, grid)

    coords = coords.flatten(2).transpose(1, 2).contiguous()       # (BV, P, 3)
    patch_valid = patch_valid.flatten(1).contiguous()             # (BV, P)

    coords = coords.reshape(B, V, num_patches, 3)
    patch_valid = patch_valid.reshape(B, V, num_patches)
    # Zero-out invalid patch coordinates so downstream stats never see leftover noise.
    coords = coords * patch_valid.unsqueeze(-1).to(coords.dtype)
    return coords, patch_valid
