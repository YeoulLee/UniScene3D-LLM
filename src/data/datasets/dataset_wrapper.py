"""Dataset wrappers for view padding and tokenization."""

import random
from pathlib import Path

import torch
from fvcore.common.registry import Registry
from transformers import AutoImageProcessor, AutoTokenizer
from torch.utils.data import Dataset

from modules.llm.prompt import (
    IGNORE_INDEX,
    load_qwen_tokenizer,
    build_input_and_labels,
    DEFAULT_SYSTEM_PROMPT,
)


DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
DATASETWRAPPER_REGISTRY.__doc__ = """ """


@DATASETWRAPPER_REGISTRY.register()
class SceneDatasetWrapper(Dataset):
    """Wrap pretraining scenes with tokenization and view sampling."""

    def __init__(self, cfg, dataset, split="train"):
        """Build the pretraining dataset wrapper."""
        self.dataset = dataset
        self.num_views = cfg.get("num_views", 32)
        model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)
        self.use_scene_cap = cfg.data.args.get("use_scene_cap", False)

    def __len__(self):
        """Return the wrapped dataset size."""
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        """Sample or pad view indices to the configured view count."""
        if num_views == self.num_views:
            return torch.arange(num_views)

        if num_views > self.num_views:
            return torch.randperm(num_views)[:self.num_views]

        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def __getitem__(self, idx):
        """Return one wrapped pretraining sample."""
        base = self.dataset[idx]
        out = {}
        view_indices = self._build_view_indices(base['point_map'].shape[0])
        view_indices_list = view_indices.tolist()
        sentence_views = [base['sentence'][i] for i in view_indices_list]
        refer_sentence_views = [base['refer_sentence'][i] for i in view_indices_list]
        encoded_input = self.tokenizer(
            [random.choice(sens) for sens in sentence_views],
            max_length=77, padding="max_length", truncation=True, return_tensors='pt'
        )

        out['txt_ids'] = encoded_input.input_ids.squeeze(0)

        refer_encoded_input = self.tokenizer(
            [random.choice(refer_sens) for refer_sens in refer_sentence_views],
            max_length=77, padding="max_length", truncation=True, return_tensors='pt'
        )

        out['ground_txt_ids'] = refer_encoded_input.input_ids.squeeze(0)

        out['images'] = self.image_processor(
            base['images'][view_indices],
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224},
            return_tensors='pt'
        )['pixel_values'].squeeze(0)

        if self.use_scene_cap:
            enc_scene = self.tokenizer(
                base['scene_cap'], max_length=248,
                padding="max_length", truncation=True, return_tensors='pt'
            )
            out['scene_txt_ids'] = enc_scene.input_ids.squeeze(0)

        out['point_map'] = base['point_map'][view_indices].contiguous().clone()
        out['scan_id'] = base['scan_id']
        return out

    def collate_fn(self, batch_list):
        """Collate wrapped pretraining samples into a batch."""
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack([value.contiguous().clone() for value in values], dim=0)
            else:
                collated[key] = values
        return collated


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapperQA(Dataset):
    """Wrap downstream QA samples with aligned view counts."""

    def __init__(self, cfg, dataset, split="train"):
        """Build the QA dataset wrapper."""
        self.dataset = dataset
        self.num_views = cfg.get("num_views", 32)
        model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)

    def __len__(self):
        """Return the wrapped dataset size."""
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        """Sample or pad view indices to the configured view count."""
        if num_views == self.num_views:
            return torch.arange(num_views)

        if num_views > self.num_views:
            return torch.randperm(num_views)[:self.num_views]

        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def __getitem__(self, idx):
        """Return one wrapped QA sample."""
        base = self.dataset[idx]
        out = {}
        for key, value in base.items():
            if torch.is_tensor(value):
                out[key] = value.contiguous().clone()
            else:
                out[key] = value

        view_source = None
        if torch.is_tensor(out.get('point_map')):
            view_source = out['point_map']
        elif torch.is_tensor(out.get('images')):
            view_source = out['images']

        if view_source is not None:
            view_indices = self._build_view_indices(view_source.shape[0])
            if torch.is_tensor(out.get('point_map')):
                out['point_map'] = out['point_map'][view_indices].contiguous().clone()
            if torch.is_tensor(out.get('images')):
                out['images'] = out['images'][view_indices].contiguous().clone()
                out['images'] = self.image_processor(
                    out['images'],
                    do_center_crop=False,
                    do_resize=True,
                    size={"height": 224, "width": 224},
                    return_tensors='pt'
                )['pixel_values'].squeeze(0).contiguous().clone()

        if torch.is_tensor(out.get('answer_label')):
            out['answer_label'] = out['answer_label'].contiguous().clone()

        return out

    def collate_fn(self, batch_list):
        """Collate wrapped QA samples into a batch."""
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack([value.contiguous().clone() for value in values], dim=0)
            else:
                collated[key] = values
        return collated


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapperLLM(Dataset):
    """Wrap QA samples for the generative LLM path.

    Produces processed multi-view images + point maps, and a Qwen chat-template token
    sequence with a single <scene> placeholder. Training builds answer-only labels;
    validation/test build a prompt-only sequence (the model generates the answer).
    """

    def __init__(self, cfg, dataset, split="train"):
        self.dataset = dataset
        self.split = split
        self.is_train = (split == "train")
        self.num_views = cfg.get("num_views", 32)

        model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)

        llm_cfg = cfg.model.get("llm", {})
        self.tokenizer, self.scene_token_id = load_qwen_tokenizer(llm_cfg.get("model_id", "Qwen/Qwen3.5-4B"))
        self.max_txt_len = int(llm_cfg.get("max_txt_len", 256))
        self.system_prompt = llm_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.enable_thinking = bool(llm_cfg.get("enable_thinking", False))
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        if num_views == self.num_views:
            return torch.arange(num_views)
        if num_views > self.num_views:
            return torch.randperm(num_views)[:self.num_views]
        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def __getitem__(self, idx):
        base = self.dataset[idx]

        # --- views: sample/pad images + point maps to a fixed count, process images to 224.
        view_indices = self._build_view_indices(base["point_map"].shape[0])
        point_map = base["point_map"][view_indices].contiguous().clone()
        images = base["images"][view_indices].contiguous().clone()
        images = self.image_processor(
            images, do_center_crop=False, do_resize=True,
            size={"height": 224, "width": 224}, return_tensors="pt",
        )["pixel_values"].squeeze(0).contiguous().clone()

        # --- text: chat-template tokenization (answer-only labels for training).
        answer = base.get("answer", "") if self.is_train else None
        enc = build_input_and_labels(
            self.tokenizer,
            situation=base["situation"],
            question=base["question"],
            answer=answer,
            system_prompt=self.system_prompt,
            max_len=self.max_txt_len,
            enable_thinking=self.enable_thinking,
        )

        out = {
            "images": images,
            "point_map": point_map,
            "input_ids": enc["input_ids"],
            "anchor_loc": base["anchor_loc"],
            "anchor_yaw": base["anchor_yaw"],
            # eval / logging metadata
            "scan_id": base["scan_id"],
            "question_id": base.get("question_id", idx),
            "sqa_type": int(base["sqa_type"]),
            "ref_answers": base.get("ref_answers", []),
            "situation": base["situation"],
            "question": base["question"],
        }
        if enc["labels"] is not None:
            out["labels"] = enc["labels"]
        return out

    def collate_fn(self, batch_list):
        collated = {}

        # Stack fixed-shape tensors.
        for key in ["images", "point_map", "anchor_loc", "anchor_yaw"]:
            collated[key] = torch.stack([s[key].contiguous().clone() for s in batch_list], dim=0)

        # Right-pad variable-length token sequences; the model re-pads via attention_mask.
        ids = [s["input_ids"] for s in batch_list]
        max_len = max(t.shape[0] for t in ids)
        B = len(ids)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, t in enumerate(ids):
            input_ids[i, : t.shape[0]] = t
            attention_mask[i, : t.shape[0]] = 1
        collated["input_ids"] = input_ids
        collated["attention_mask"] = attention_mask

        if "labels" in batch_list[0]:
            labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
            for i, s in enumerate(batch_list):
                t = s["labels"]
                labels[i, : t.shape[0]] = t
            collated["labels"] = labels

        # Keep python-side metadata as lists.
        collated["sqa_type"] = torch.tensor([s["sqa_type"] for s in batch_list], dtype=torch.long)
        for key in ["scan_id", "question_id", "ref_answers", "situation", "question"]:
            collated[key] = [s[key] for s in batch_list]
        return collated
