"""Trainer for the generative UniScene3D-LLM path (SQA3D).

Differs from DefaultTrainer in two ways:
  * backward() respects gradient accumulation (steps only on the accumulation boundary),
    which matters for full fine-tuning a 4B LLM with micro-batches.
  * eval/test generate answer strings and score them with the generative EM evaluator,
    gathering per-sample prediction records across processes.

NOTE (ZeRO-3): generation makes many forward passes through the (sharded) LLM. If you run
DeepSpeed ZeRO-3, ensure params are gathered for generation (e.g. evaluate with a lower zero
stage, or wrap generation in deepspeed.zero.GatheredParameters). Single-GPU / ZeRO<=2 eval
works as-is.
"""

from tqdm import tqdm
import torch

from trainer.build import TRAINER_REGISTRY
from trainer.default_trainer import DefaultTrainer
from common.misc import gather_object


@TRAINER_REGISTRY.register()
class LLMTrainer(DefaultTrainer):
    """Full fine-tune trainer with generative SQA3D evaluation."""

    def backward(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if self.grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def _collect_predictions(self, loader, desc="eval"):
        """Run generation over a loader and return per-sample prediction records."""
        records = []
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=desc)
        for data_dict in loader:
            data_dict = self.forward(data_dict, mode="qa")
            preds = data_dict["output_text"]
            for i in range(len(preds)):
                records.append({
                    "pred": preds[i],
                    "ref_answers": data_dict["ref_answers"][i],
                    "sqa_type": int(data_dict["sqa_type"][i].item()),
                    "scan_id": data_dict["scan_id"][i],
                    "question_id": data_dict["question_id"][i],
                    "situation": data_dict["situation"][i],
                    "question": data_dict["question"][i],
                })
            pbar.update(1)
        # Gather string/dict records from all processes onto every process.
        records = gather_object(records)
        return records

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        records = self._collect_predictions(self.data_loaders["val"], desc=f"[eval {epoch + 1}]")
        self.accelerator.wait_for_everyone()
        is_best = False
        if self.accelerator.is_main_process:
            is_best, results = self.evaluator.evaluate(records, split="val")
            if is_best:
                self.best_metric = results["target_metric"]
            self.log(results, mode="val")
        # Broadcast so every rank agrees whether to save best.pth (all ranks must call save).
        return self._broadcast_flag(is_best)

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        records = self._collect_predictions(self.data_loaders["test"], desc="[test]")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            _, results = self.evaluator.evaluate(records, split="test")
            self.log(results, mode="test")
            return results
        return None

    def run(self):
        """Training loop with DeepSpeed-safe checkpointing.

        Unlike DefaultTrainer.run, save_state() is called by ALL ranks (it is a collective
        under ZeRO-3: sharded params are gathered across ranks), and is_best is already
        broadcast by eval_step, so every rank takes the same save path. Guarding the save with
        is_main_process (as DefaultTrainer does) deadlocks under DeepSpeed.
        """
        if self.mode == "train":
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.set_downstream_mode()
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])

            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] eval done, is_best={is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                # Collective save: every rank participates (do NOT guard with is_main_process).
                self.save("latest.pth")
                if is_best:
                    self.save("best.pth")
                if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                    self.save(f"ckpt_{epoch + 1}.pth")
                self.accelerator.wait_for_everyone()

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()
