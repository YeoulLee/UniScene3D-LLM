"""Evaluation logic for the SQA3D task."""

import os
import json
import collections
from pathlib import Path

import numpy as np
import torch

from data.data_utils import SQA3DAnswer, clean_answer
from evaluator.common.build import EVALUATOR_REGISTRY
import re

@EVALUATOR_REGISTRY.register()
class SQA3DEval():
    """Evaluator for SQA3D answer classification."""

    # 0: what, 1: is, 2: how, 3: can, 4: which, 5: others
    def __init__(self, cfg, task_name):
        """Load answer metadata and initialize running metrics."""
        self.eval_dict = {
            'target_metric': [], 'ans1_acc': [], 'ans10_acc': [], 'non_color_ans1_acc': [], 'non_color_ans10_acc': [],
            'type0_acc': [], 'type1_acc': [], 'type2_acc': [],
            'type0_acc': [], 'type1_acc': [], 'type2_acc': [],
            'type3_acc': [], 'type4_acc': [], 'type5_acc': []
        }
        # run
        self.total_count = 0
        self.non_color_total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        self.best_result = -np.inf
        self.base_dir = cfg.data.scan_family_base

        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'annotations/sqa3d/answer_dict.json'), encoding='utf-8')
        )[0]
        
        color_terms = [
            "orange", "pink", "maroon", "grey", "gray", "purple",
            "red", "yellow", "brown", "blue", "green", "silver", "gold",
            "tan", "turquoise", "beige", "white", "black", "chocolate",
            "multicolored",
            "black and red", "yellow and orange", "black white",
            "light brown", "dark brown",
            "balck", "white"  
        ]

        # Collect all color-related answers with their ids
        self.color_ids = [answer_data[name] for name in color_terms]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_cands = answer_counter.keys()
        self.answer_vocab = SQA3DAnswer(answer_cands)

        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / task_name
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, data_dict):
        """Accumulate one batch of SQA3D predictions."""
        metrics = self.batch_metrics(data_dict)
        batch_count = metrics['total_count']
        self.total_count += batch_count
        non_color_batch_count = metrics['non_color_total_count']
        self.non_color_total_count += non_color_batch_count
        for key in metrics:
            if 'type' in key and 'count' in key:
                self.type_count[key] += metrics[key]

        if self.save:
            for i in range(metrics["total_count"]):
                self.eval_results.append({
                    # vision
                    "source": data_dict['source'][i],
                    "scan_id": data_dict['scan_id'][i],
                    "anchor": data_dict['anchor_locs'][i],
                    'anchor_ort': data_dict['anchor_orientation'][i],
                    # language
                    "instruction": data_dict['prompt_after_obj'][i],
                    "response_gt": data_dict['answer_list'][i].split('[answer_seq]'),
                    "response_pred": data_dict['output_text'][i]
                })

        # save eval dict
        for key in self.eval_dict.keys():
            if 'type' in key:
                self.eval_dict[key].append(float(metrics[key]) * metrics['type' + key[4] + '_count'])
            elif 'non_color' not in key:
                self.eval_dict[key].append(float(metrics[key]) * batch_count)
            else:
                self.eval_dict[key].append(float(metrics[key]) * non_color_batch_count)

    def batch_metrics(self, data_dict):
        """Compute batch-level answer accuracy metrics."""
        metrics = {}

        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        
        correct1 = 0
        correct10 = 0
        
        non_color_correct1 = 0
        non_color_correct10 = 0
        total_non_color = 0
        
        correct_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        count_type = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 1e-10}
        for i in range(data_dict['answer_label'].shape[0]):
            gt_id = data_dict['answer_label'][i].argmax().item()
            if gt_id not in self.color_ids:
                total_non_color += 1
            
            count_type[data_dict['sqa_type'][i].item()] += 1
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                if choice_1[i] not in self.color_ids:
                    non_color_correct1 += 1
                correct1 += 1
                correct_type[data_dict['sqa_type'][i].item()] += 1
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    if choice_10[i,j] not in self.color_ids:
                        non_color_correct10 += 1
                    correct10 += 1
                    break
                
        metrics['ans1_acc'] = correct1 / float(len(choice_1))
        metrics['ans10_acc'] = correct10 / float(len(choice_1))
        
        metrics['non_color_ans1_acc'] = non_color_correct1 / float(total_non_color)
        metrics['non_color_ans10_acc'] = non_color_correct10 / float(total_non_color)

        # question type acc
        for key in count_type.keys():
            metrics['type' + str(key) + '_acc'] = correct_type[key] / count_type[key]
            metrics['type' + str(key) + '_count'] = count_type[key]

        metrics['target_metric'] = metrics['ans1_acc']
        metrics["total_count"] = data_dict["answer_scores"].shape[0]
        metrics["non_color_total_count"] = total_non_color
        return metrics

    def reset(self):
        """Reset the running evaluation state."""
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.non_color_total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10, 
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        if self.save:
            self.eval_results = []

    def record(self, split='val'):
        """Finalize metrics and optionally save predictions."""
        # record
        for k, v in self.eval_dict.items():
            if k == "answer_top10":
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count['type' + k[4] + '_count']
            elif 'non_color' not in k:
                self.eval_dict[k] = sum(v) / self.total_count
                print(k, 'overall', sum(v), self.total_count)
            elif 'non_color' in k:
                self.eval_dict[k] = sum(v) / self.non_color_total_count
                print(k, 'non_color', sum(v), self.non_color_total_count)

        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False

        if self.save and (is_best or split == 'test'):
            torch.save(self.eval_results, str(self.save_dir / 'results.pt'))

        return is_best, self.eval_dict


@EVALUATOR_REGISTRY.register()
class SQA3DLLMEval():
    """Generative exact-match evaluator for SQA3D.

    Scores model-generated answer strings against the ground-truth answer set using the
    repo's clean_answer normalization, and reports overall EM plus per-question-type EM
    (0: what, 1: is, 2: how, 3: can, 4: which, 5: others).
    """

    TYPE_NAMES = {0: "what", 1: "is", 2: "how", 3: "can", 4: "which", 5: "others"}

    def __init__(self, cfg, accelerator):
        self.accelerator = accelerator
        self.best_result = -np.inf
        self.save = cfg.eval.get("save", False)
        if self.save:
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "sqa3d_llm"
            self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _match(pred, refs):
        """Return 1 if the normalized prediction matches any normalized reference answer."""
        p = clean_answer(pred)
        ref_set = {clean_answer(r) for r in refs}
        return 1.0 if p in ref_set else 0.0

    def evaluate(self, records, split="val"):
        """Compute EM metrics over gathered per-sample prediction records."""
        # Deduplicate in case gather replicated records across processes.
        seen, unique = set(), []
        for r in records:
            key = (r.get("scan_id"), r.get("question_id"), r.get("question"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        records = unique

        total = max(len(records), 1)
        correct = 0.0
        type_correct = {t: 0.0 for t in self.TYPE_NAMES}
        type_count = {t: 0 for t in self.TYPE_NAMES}

        for r in records:
            hit = self._match(r["pred"], r["ref_answers"])
            correct += hit
            t = int(r["sqa_type"])
            type_correct[t] += hit
            type_count[t] += 1

        results = {"em_acc": correct / total, "num_samples": float(len(records))}
        for t, name in self.TYPE_NAMES.items():
            results[f"type_{t}_{name}_acc"] = type_correct[t] / max(type_count[t], 1)
        results["target_metric"] = results["em_acc"]

        is_best = results["target_metric"] > self.best_result
        if is_best:
            self.best_result = results["target_metric"]
        results["best_result"] = self.best_result

        if self.save and self.accelerator.is_main_process and (is_best or split == "test"):
            with (self.save_dir / f"results_{split}.json").open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

        return is_best, results

    def reset(self):
        return

