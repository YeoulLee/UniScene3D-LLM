"""Trainer registry and trainer setup logic."""

import glob
from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedType
from fvcore.common.registry import Registry
import torch
import wandb

import common.misc as misc
from data.build import build_dataloader
from evaluator.common.build import build_eval
from model.build import build_model
from optim.build import build_optim
from safetensors.torch import load_file

TRAINER_REGISTRY = Registry("Trainer")


class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.exp_dir.parent.name.replace(f'{cfg.name}', '').lstrip('_')}/{cfg.exp_dir.name}"
        self.epoch = 0
        self.best_result = -np.inf

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.get("flag", False)
        self.hard_debug = cfg.debug.get("hard_debug", False)
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        self.global_step = 0

        self.exp_tracker = Tracker(cfg)
        wandb_args = {"entity": cfg.logger.entity, "id": cfg.logger.run_id, "resume": cfg.resume}
        if not cfg.logger.get('autoname'):
            wandb_args["name"] = self.exp_tracker.exp_name
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        gradient_accumulation_steps = cfg.solver.get("gradient_accumulation_steps", 1)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs,
        )

        if not self.hard_debug:
            self.accelerator.init_trackers(
                project_name=cfg.name if not self.debug else "Debug",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if not cfg.resume else None,
                init_kwargs={"wandb": wandb_args},
            )

        print(OmegaConf.to_yaml(cfg))

        if self.mode == "pretrain":
            keys = ["pretrain"]
        else:
            keys = ["train", "val", "test"]

        # Build data first so model and optimizer can use dataloader sizes.
        self.data_loaders = {key: build_dataloader(cfg, split=key) for key in keys}
        self.model = build_model(cfg)
        self.epochs = cfg.solver.epochs

        if self.mode == "test":
            total_steps = 1
        else:
            total_steps = (len(self.data_loaders[self.mode]) * self.epochs) // gradient_accumulation_steps
        self.loss, self.optimizer, self.scheduler = build_optim(
            cfg,
            self.model.get_opt_params(),
            total_steps=total_steps,
            accelerator=self.accelerator,
        )

        if self.mode == "pretrain":
            self.evaluator = None
        else:
            if misc.rgetattr(cfg, "eval.pass_kwargs", False):
                kwargs = {"dataloaders": self.data_loaders}
            else:
                kwargs = {}
            self.evaluator = build_eval(cfg, self.accelerator, **kwargs)

        self.total_steps = 1 if self.mode == "test" else len(self.data_loaders[self.mode]) * self.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        ema = [0.996, 1.0]
        ipe_scale = 1.0
        self.momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (self.total_steps * self.epochs * ipe_scale)
            for i in range(int(self.total_steps * self.epochs * ipe_scale) + 1)
        )

        if cfg.get('pretrain_ckpt_path'):
            self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
            self.load_pretrain()
            # Keep the point-map embedding in sync with the RGB patch embedding if requested.
            if hasattr(self.model, "sync_geo_embedding_from_patch_after_load"):
                self.model.sync_geo_embedding_from_patch_after_load()
            if hasattr(self.model, "pm_encoder"):
                self.model.pm_encoder.load_state_dict(self.model.pm_encoder.state_dict())

        # Let Accelerate wrap the model, optimizer, and dataloaders for the chosen backend.
        # DeepSpeed forbids preparing two nn.Modules with one Accelerator; self.loss is an
        # nn.Module (parameter-free for lm_loss), so under DeepSpeed we prepare only the model
        # and just move the loss to the right device.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            self.loss = self.loss.to(self.accelerator.device)
        else:
            self.model, self.loss, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.loss, self.optimizer, self.scheduler
            )
        for name, loader in self.data_loaders.items():
            if isinstance(loader, list):
                loader = self.accelerator.prepare(*loader)
            else:
                loader = self.accelerator.prepare(loader)
            self.data_loaders[name] = loader
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        self.ckpt_path = Path(cfg.ckpt_path) if cfg.get("ckpt_path") else Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if cfg.resume:
            self.resume()
        elif self.mode == "test" and cfg.get("test_state_dict"):
            # Robust evaluation path: load a CONSOLIDATED state_dict (e.g. produced by
            # deepspeed's zero_to_fp32.py, or the gathered fp16 model). Avoids accelerate's
            # deepspeed load_state, so it works regardless of the #GPUs used for training.
            self._load_test_state_dict(cfg.test_state_dict)
        elif self.mode == "test" and cfg.get("ckpt_path"):
            # Standalone evaluation from an accelerate/deepspeed checkpoint directory, bypassing
            # run.py's resume branch (which reloads the saved config.yaml and forces mode=train).
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"test ckpt_path does not exist: {self.ckpt_path}")
            self.accelerator.print(f"📂 Loading test checkpoint from: {self.ckpt_path}")
            self.accelerator.load_state(str(self.ckpt_path))

    def forward(self, data_dict):
        return self.model(data_dict)

    def update_ema(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            model_context = self.model.module.context_model if hasattr(self.model, 'module') else self.model.context_model
            model_target = self.model.module.target_model if hasattr(self.model, 'module') else self.model.target_model

            for param_q, param_k in zip(model_context.parameters(), model_target.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def backward(self, loss):
        self.accelerator.backward(loss)

        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach()) for p in self.model.parameters() if p.grad is not None
        ]))
        print(f"grad_norm={total_norm.item():.2f}")

        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        if self.accelerator.sync_gradients:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def _broadcast_flag(self, value):
        """Make a main-process boolean consistent across all ranks (gather-max).

        Only the main process computes metrics, so a flag like is_best is real there and 0
        elsewhere; the max over ranks equals the main process's value on every rank. Identical
        control flow across ranks is required to avoid collective deadlocks at checkpoint save
        (DeepSpeed ZeRO-3 save_state gathers sharded params and needs every rank to call it).
        """
        t = torch.tensor([1.0 if value else 0.0], device=self.accelerator.device)
        return bool(self.accelerator.gather(t).max().item() > 0.5)

    def log(self, results, mode="train"):
        if not self.hard_debug:
            log_dict = {}
            for key, val in results.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"{mode}/lr/group_{i}"] = lr
            self.accelerator.log(log_dict, step=self.global_step)

    def save(self, name):
        misc.make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def save_consolidated(self, name):
        """Save a merged (ZeRO-3-gathered) model state_dict for easy eval/export.

        Unlike save_state (full training state; sharded as zero_pp_rank_* files that accelerate
        cannot reload for plain inference), this writes the consolidated model weights via
        accelerator.save_model (safetensors). Load it later with +test_state_dict=<dir>, on any
        #GPUs, without zero_to_fp32. Collective: every rank must call it under DeepSpeed.
        """
        out_dir = self.ckpt_path.parent / name
        misc.make_dir(out_dir)
        self.accelerator.save_model(self.model, str(out_dir))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")

    def _load_test_state_dict(self, sd_path):
        """Load a consolidated full-model state_dict for evaluation (ZeRO-3-partition safe).

        Accepts .safetensors, or a torch-saved .pt/.bin (optionally wrapping the weights under
        a 'state_dict'/'module' key). Strips a leading 'module.' from keys, then loads with the
        same partition-aware path used for the pretrained encoder.
        """
        from safetensors.torch import load_file
        p = Path(sd_path)
        if not p.exists():
            raise FileNotFoundError(f"test_state_dict does not exist: {sd_path}")

        sd = {}
        if p.is_dir():
            # zero_to_fp32 may emit a folder of sharded safetensors / bins; merge them all.
            files = sorted(glob.glob(str(p / "*.safetensors"))) or \
                    sorted(glob.glob(str(p / "*.bin"))) or \
                    sorted(glob.glob(str(p / "*.pt")))
            if not files:
                raise FileNotFoundError(f"No weight shards (*.safetensors/*.bin/*.pt) in {sd_path}")
            for f in files:
                part = load_file(f, device="cpu") if f.endswith(".safetensors") else torch.load(f, map_location="cpu")
                sd.update(part)
        elif sd_path.endswith(".safetensors"):
            sd = load_file(sd_path, device="cpu")
        else:
            sd = torch.load(sd_path, map_location="cpu")
            if isinstance(sd, dict):
                for key in ("state_dict", "module", "model"):
                    if key in sd and isinstance(sd[key], dict):
                        sd = sd[key]
                        break
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.accelerator.print(f"📂 Loading test state_dict from: {sd_path} ({len(sd)} tensors)")
        self._load_weights_into_model(sd)

    def load_pretrain(self):
        self.accelerator.print(f"📂 Loading pretrained weights from: {str(self.pretrain_ckpt_path)}")
        model_weight_path_pattern = str(self.pretrain_ckpt_path / "model*.safetensors")
        model_weight_paths = glob.glob(model_weight_path_pattern)

        if len(model_weight_paths) == 0:
            raise FileNotFoundError(f"❌ Cannot find any .safetensors file in {str(self.pretrain_ckpt_path)}")

        weights = {}
        for model_weight_path in model_weight_paths:
            # Merge shard files into one state dict before loading.
            weights.update(load_file(model_weight_path, device="cpu"))

        self._load_weights_into_model(weights)

        vision_model_keys = {
            key for key, _ in self.model.named_parameters()
            if key.startswith("pm_encoder.vision_model.")
        }
        loaded_vision_model_keys = vision_model_keys.intersection(weights.keys())

        if vision_model_keys and loaded_vision_model_keys == vision_model_keys:
            self.accelerator.print(f"✅ Entire UniScene3D vision encoder is loaded.")

    def _load_weights_into_model(self, weights):
        """Load a (full-shape) state dict, robust to DeepSpeed ZeRO-3 parameter partitioning.

        Under ZeRO-3 (zero.Init), params are sharded at construction so their local shape is
        e.g. torch.Size([0]); a plain load_state_dict then errors on shape mismatch. In that
        case we gather each param with deepspeed.zero.GatheredParameters, copy the checkpoint
        value on rank 0, and let the context re-partition/broadcast on exit. Buffers are not
        partitioned and are copied directly. Without ZeRO-3 this falls back to load_state_dict.
        """
        # Use the UNWRAPPED model so param names match the checkpoint: after prepare the
        # DeepSpeed/DDP wrapper adds a 'module.' prefix to named_parameters(), which would make
        # every key miss the (unprefixed) consolidated checkpoint. unwrap_model is a no-op
        # pre-prepare (used by load_pretrain during training).
        model = self.accelerator.unwrap_model(self.model)
        # Also tolerate a leading 'module.' on the checkpoint side.
        weights = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in weights.items()}

        params = list(model.named_parameters())
        is_zero3 = any(hasattr(p, "ds_id") for _, p in params)

        if not is_zero3:
            incompatible = model.load_state_dict(weights, strict=False)
            n_missing = len(getattr(incompatible, "missing_keys", []))
            self.accelerator.print(f"📦 Loaded weights: {len(params) - n_missing}/{len(params)} "
                                   f"params matched ({n_missing} missing).")
            return

        import deepspeed  # local import: only needed on the ZeRO-3 path
        from collections import Counter

        missing, missing_trainable = [], []
        for name, p in params:
            if name not in weights:
                missing.append(name)
                if p.requires_grad:
                    missing_trainable.append(name)
                continue
            with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                if self.accelerator.is_main_process:
                    p.data.copy_(weights[name].to(device=p.device, dtype=p.dtype))
        # Buffers (e.g. logit-scale-adjacent or non-persistent) are replicated, not sharded.
        for name, b in model.named_buffers():
            if name in weights:
                b.data.copy_(weights[name].to(device=b.device, dtype=b.dtype))
        copied = len(params) - len(missing)
        self.accelerator.print(f"📦 ZeRO-3 partition-safe load: copied {copied}/{len(params)} "
                               f"params ({len(missing)} not in checkpoint, left as-is).")
        if missing:
            by_module = Counter(n.split(".")[0] for n in missing)
            self.accelerator.print(f"   missing by module: {dict(by_module)}")
        if copied == 0:
            raise RuntimeError(
                "Loaded 0 params — checkpoint keys do not match the model. Check that the "
                "model switches match training and that the consolidated dir is correct."
            )
        if missing_trainable:
            # Trainable params (projector/LLM) MUST come from the checkpoint; frozen pm_encoder
            # is fine to miss here (it is restored by load_pretrain).
            by_mod = Counter(n.split(".")[0] for n in missing_trainable)
            raise RuntimeError(
                f"{len(missing_trainable)} TRAINABLE params missing from the checkpoint "
                f"{dict(by_mod)} — eval would use untrained weights. The consolidated checkpoint "
                "likely does not contain the trained projector/LLM weights."
            )

    def save_func(self, path):
        self.accelerator.save_state(path)

def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
