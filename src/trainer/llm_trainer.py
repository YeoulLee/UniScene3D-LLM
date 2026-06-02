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
        if self.accelerator.is_main_process:
            is_best, results = self.evaluator.evaluate(records, split="val")
            if is_best:
                self.best_metric = results["target_metric"]
            self.log(results, mode="val")
            return is_best
        return False

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        records = self._collect_predictions(self.data_loaders["val"], desc="[test]")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            _, results = self.evaluator.evaluate(records, split="test")
            self.log(results, mode="test")
            return results
        return None
