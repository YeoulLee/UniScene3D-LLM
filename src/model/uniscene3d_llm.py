"""UniScene3D-LLM: 3D-grounded generative VQA model for SQA3D.

Architecture (see configs/finetune/sqa3d_llm.yaml):

    images + point_map ──► FG-CLIP (frozen) ──► patch tokens (B,V,P,D)
    point_map ──► coord_pool ──► per-patch 3D coords (B,V,P,3) + valid
                              ──► frame(world|ego) + per-scene normalization
    tokens + sinusoidal 3D PE ──► TokenReducer(Identity) ──► Projector(MLP)
                              ──► spliced into Qwen at the <scene> token ──► LM

Training returns the next-token LM loss over answer tokens (data_dict['lm_loss']).
Evaluation generates the answer string (data_dict['output_text']).
"""

from pathlib import Path

import torch
from torch.amp import autocast

from common.misc import build_fgclip_model_from_local_code_with_hf_weights
from model.build import MODEL_REGISTRY, BaseModel
from optim.utils import no_decay_param_group
from modules.lift3d import (
    pool_pointmap_to_patches,
    apply_coord_frame,
    sinusoidal_pos_embed_3d,
    build_token_reducer,
    Projector,
)
from modules.llm import QwenLLM


@MODEL_REGISTRY.register()
class UniScene3DLLM(BaseModel):
    """FG-CLIP 3D scene encoder + global 3D PE + Qwen LLM for SQA3D."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        mcfg = cfg.model

        # --- FG-CLIP point-map encoder (frozen). Named 'pm_encoder' so the UniScene3D
        # pretrained checkpoint (pm_encoder.vision_model.*) loads via BaseTrainer.load_pretrain.
        model_root = str(Path(__file__).resolve().parents[1] / "fg-clip")
        fgclip_repo_id = mcfg.get("fgclip_repo_id", None)
        if fgclip_repo_id is None:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root)
        else:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root, repo_id=fgclip_repo_id)

        # --- 3D lifting config
        self.vision_feature = mcfg.get("vision_feature", "projected")  # projected(512) | penultimate(768)
        assert self.vision_feature in ("projected", "penultimate")
        self.feature_dim = 512 if self.vision_feature == "projected" else 768
        self.coord_frame = mcfg.get("coord_frame", "world")            # world | ego
        pe_cfg = mcfg.get("pos_embed", {})
        self.pos_normalize = pe_cfg.get("normalize", "none")           # none(paper) | scene_bbox | fixed_scale
        self.pos_temperature = float(pe_cfg.get("temperature", 10000.0))
        self.pos_fixed_scale = float(pe_cfg.get("fixed_scale", 10.0))

        # --- token reducer (Identity now; registry plug-in later)
        self.reducer = build_token_reducer(mcfg.get("reducer", {"name": "Identity"}))

        # --- LLM
        llm_cfg = mcfg.get("llm", {})
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 32))
        dtype = torch.bfloat16 if str(llm_cfg.get("torch_dtype", "bfloat16")) == "bfloat16" else torch.float16
        self.llm = QwenLLM(
            model_id=llm_cfg.get("model_id", "Qwen/Qwen3.5-4B"),
            torch_dtype=dtype,
            attn_implementation=llm_cfg.get("attn_implementation", None),
            gradient_checkpointing=bool(llm_cfg.get("gradient_checkpointing", False)),
        )
        # Expose the LLM config as `.config` so Accelerate/DeepSpeed can auto-fill ZeRO-3
        # bucket sizes from hidden_size (this wrapper otherwise has no .config).
        self.config = self.llm.model.config

        # --- projector (feature_dim -> llm hidden)
        proj_cfg = mcfg.get("projector", {})
        self.projector = Projector(
            in_dim=self.feature_dim,
            llm_hidden=self.llm.hidden_size,
            hidden_dim=proj_cfg.get("hidden_dim", None),
            depth=int(proj_cfg.get("depth", 2)),
        )

        self.set_downstream_mode()

    # ------------------------------------------------------------------ encoder
    def _dense_features(self, color_pm):
        """Per-patch FG-CLIP features (B*V, P, feature_dim) with the dense-localization trick.

        Mirrors FGCLIPModel.get_image_dense_features but exposes the pre-projection
        penultimate features (768) when vision_feature == 'penultimate'.
        """
        vm = self.pm_encoder.vision_model
        vision_outputs = vm(pixel_values=color_pm, output_hidden_states=True, return_dict=True)
        feature_map = vision_outputs.hidden_states[-2]
        feature_map = self.pm_encoder.forward_without_attn(feature_map)[:, 1:]  # drop CLS
        feature_map = vm.post_layernorm(feature_map)
        if self.vision_feature == "projected":
            feature_map = self.pm_encoder.visual_projection(feature_map)
        return feature_map

    def _encode_patches(self, images, point_map):
        """Encode colored point maps into per-view patch tokens (B, V, P, D)."""
        B, V, C, H, W = images.shape
        images = images.reshape(B * V, C, H, W).contiguous().float()
        pm = point_map.reshape(B * V, C, H, W).contiguous().float()
        color_pm = torch.cat([images, pm], dim=1)  # 6-channel colored point map
        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                feat = self._dense_features(color_pm)  # (B*V, P, D)
        P, D = feat.shape[1], feat.shape[2]
        return feat.reshape(B, V, P, D)

    # ------------------------------------------------------------------ forward
    def forward(self, data_dict, mode="qa"):
        images = data_dict["images"]        # (B, V, 3, H, W)
        point_map = data_dict["point_map"]  # (B, V, 3, H, W)
        B, V = images.shape[0], images.shape[1]

        feats = self._encode_patches(images, point_map)        # (B, V, P, D)
        P = feats.shape[2]

        coords, valid = pool_pointmap_to_patches(point_map, P)  # (B,V,P,3), (B,V,P)
        coords = apply_coord_frame(
            coords, valid, self.coord_frame,
            normalize=self.pos_normalize,
            anchor_loc=data_dict.get("anchor_loc"),
            anchor_yaw=data_dict.get("anchor_yaw"),
            fixed_scale=self.pos_fixed_scale,
        )

        pe = sinusoidal_pos_embed_3d(coords, feats.shape[-1], temperature=self.pos_temperature)
        feats = feats + pe.to(feats.dtype)

        tokens = feats.reshape(B, V * P, feats.shape[-1])
        coords_f = coords.reshape(B, V * P, 3)
        valid_f = valid.reshape(B, V * P)
        tokens, coords_f, valid_f = self.reducer(tokens, coords_f, valid_f)

        with autocast("cuda", dtype=torch.bfloat16):
            visual = self.projector(tokens)  # (B, N, llm_hidden)

        if self.training:
            loss = self.llm(
                visual,
                data_dict["input_ids"],
                data_dict["attention_mask"],
                data_dict["labels"],
            )
            data_dict["lm_loss"] = loss
        else:
            data_dict["output_text"] = self.llm.generate(
                visual,
                data_dict["input_ids"],
                data_dict["attention_mask"],
                max_new_tokens=self.max_new_tokens,
            )
        return data_dict

    # ------------------------------------------------------------------ modes / params
    def set_downstream_mode(self):
        """Freeze the FG-CLIP encoder; train projector + LLM."""
        for p in self.pm_encoder.parameters():
            p.requires_grad = False
        self.pm_encoder.eval()
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.llm.parameters():
            p.requires_grad = True

    def sync_geo_embedding_from_patch_after_load(self):
        """No-op hook (only meaningful in pretrain mode); kept for BaseTrainer compatibility."""
        return

    def get_opt_params(self):
        """Full fine-tune: all trainable params (projector + LLM) at solver.lr."""
        lr = self.cfg.solver.lr
        params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        return no_decay_param_group(params, lr)
