"""3D lifting modules: turn FG-CLIP patch tokens into 3D-positioned LLM tokens.

Pipeline order (see UniScene3DLLM.forward):
    patch tokens (B,V,P,D) + point_map (B,V,3,H,W)
      -> coord_pool   : per-patch 3D coordinate (B,V,P,3) + valid mask
      -> frame        : world | ego coordinate frame + per-scene normalization
      -> pos_embed_3d : sinusoidal 3D PE, added to tokens (no learnable params)
      -> token_reducer: Identity now (registry plug-in for voxel/pixel-shuffle later)
      -> projector    : MLP D -> llm_hidden
"""

from .coord_pool import pool_pointmap_to_patches
from .frame import apply_coord_frame, normalize_coords
from .pos_embed_3d import sinusoidal_pos_embed_3d
from .token_reducer import TOKEN_REDUCER_REGISTRY, build_token_reducer
from .projector import Projector

__all__ = [
    "pool_pointmap_to_patches",
    "apply_coord_frame",
    "normalize_coords",
    "sinusoidal_pos_embed_3d",
    "TOKEN_REDUCER_REGISTRY",
    "build_token_reducer",
    "Projector",
]
