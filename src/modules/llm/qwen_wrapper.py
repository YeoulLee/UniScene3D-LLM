"""Qwen wrapper: splice projected scene tokens into the text stream and run the LLM.

The text sequence carries exactly one ``<scene>`` token per sample (see prompt.py). At
forward/generate time we replace that single token embedding with the N projected visual
embeddings, rebuild the attention mask and labels, and call the Qwen causal LM on
``inputs_embeds``. Training returns the LM loss; generation returns decoded answer strings.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .prompt import IGNORE_INDEX, load_qwen_tokenizer


class QwenLLM(nn.Module):
    """Causal Qwen LM that consumes interleaved text + projected scene tokens."""

    def __init__(self, model_id: str, torch_dtype=torch.bfloat16, attn_implementation=None,
                 gradient_checkpointing=False, lora_cfg=None):
        super().__init__()
        self.model_id = model_id
        self.tokenizer, self.scene_token_id = load_qwen_tokenizer(model_id)

        load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # Account for the added <scene> token (must happen before LoRA wraps the model).
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Optional LoRA: freeze the base LM and train only low-rank adapters.
        self.use_lora = bool(lora_cfg and lora_cfg.get("enabled", False))
        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=list(lora_cfg.get("target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
            )
            self.model = get_peft_model(self.model, peft_config)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # LoRA + checkpointing: the frozen base needs input grads for backprop to flow.
            if self.use_lora and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.config.use_cache = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _embed(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def merge_visual(self, input_ids, attention_mask, visual_embeds, labels=None, pad_side="right"):
        """Replace each sample's <scene> token with its N projected scene embeddings.

        Args:
            input_ids: (B, L) padded token ids, exactly one <scene> per row.
            attention_mask: (B, L) 1 for real tokens.
            visual_embeds: (B, N, H) projected scene tokens.
            labels: (B, L) or None. Visual positions are filled with IGNORE_INDEX.
            pad_side: "right" for training, "left" for generation.

        Returns:
            inputs_embeds (B, L', H), attention_mask (B, L'), labels (B, L') or None.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        embed_w = self.model.get_input_embeddings().weight
        if visual_embeds is not None:
            visual_embeds = visual_embeds.to(embed_w.dtype)

        merged_embeds, merged_labels, lengths = [], [], []
        for b in range(B):
            keep = attention_mask[b].bool()
            ids_b = input_ids[b][keep]
            txt_emb = self._embed(ids_b)  # (Lb, H)
            lbl_b = labels[b][keep] if labels is not None else None

            scene_pos = (ids_b == self.scene_token_id).nonzero(as_tuple=False)
            if visual_embeds is None or scene_pos.numel() == 0:
                # Text-only (use_vision=False ablation) or missing scene token:
                # keep <scene> as a single placeholder token, no splice.
                emb_b = txt_emb
            else:
                p = int(scene_pos[0, 0])
                emb_b = torch.cat([txt_emb[:p], visual_embeds[b], txt_emb[p + 1:]], dim=0)
                if labels is not None:
                    n = visual_embeds.shape[1]
                    ignore = torch.full((n,), IGNORE_INDEX, dtype=lbl_b.dtype, device=device)
                    lbl_b = torch.cat([lbl_b[:p], ignore, lbl_b[p + 1:]], dim=0)
            merged_embeds.append(emb_b)
            merged_labels.append(lbl_b)
            lengths.append(emb_b.shape[0])

        max_len = max(lengths)
        H = merged_embeds[0].shape[-1]
        out_emb = torch.zeros(B, max_len, H, dtype=merged_embeds[0].dtype, device=device)
        out_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        out_lbl = None
        if labels is not None:
            out_lbl = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=device)

        for b in range(B):
            L = lengths[b]
            if pad_side == "left":
                out_emb[b, max_len - L:] = merged_embeds[b]
                out_mask[b, max_len - L:] = 1
                if labels is not None:
                    out_lbl[b, max_len - L:] = merged_labels[b]
            else:
                out_emb[b, :L] = merged_embeds[b]
                out_mask[b, :L] = 1
                if labels is not None:
                    out_lbl[b, :L] = merged_labels[b]

        return out_emb, out_mask, out_lbl

    def forward(self, visual_embeds, input_ids, attention_mask, labels):
        """Training forward: returns the scalar LM loss."""
        inputs_embeds, attn, lbl = self.merge_visual(
            input_ids, attention_mask, visual_embeds, labels=labels, pad_side="right",
        )
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=lbl)
        return out.loss

    @torch.no_grad()
    def generate(self, visual_embeds, input_ids, attention_mask, max_new_tokens=32, **gen_kwargs):
        """Greedy/sampled generation; returns a list of decoded answer strings."""
        inputs_embeds, attn, _ = self.merge_visual(
            input_ids, attention_mask, visual_embeds, labels=None, pad_side="left",
        )
        # Re-enable the KV cache for fast decoding (training disables it for checkpointing).
        prev_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True
        gen = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        self.model.config.use_cache = prev_use_cache
        # With inputs_embeds, generate returns only the newly generated token ids.
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]
