from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
import copy

import src.api.task as task_api
import src.api.result as result_api
import src.api.model as model_api

torch.set_grad_enabled(False)


def make_input_ids(
    num_samples,
    seg_len,
    rep,
    vocab_size,
    prepend_bos=False,
    bos=None,
) -> np.ndarray:
    sample_int = np.random.randint(
        low=0, high=vocab_size, size=num_samples * seg_len
    ).reshape(num_samples, seg_len)
    sample_int = np.concatenate(tuple([sample_int] * rep), axis=1)

    if prepend_bos:
        sample_int = np.hstack([bos * np.ones((num_samples, 1), dtype=int), sample_int])

    input_ids = np.array(sample_int, dtype=np.int32)

    return input_ids


class CopyingTask(task_api.Task):
    def __init__(self, seg_len, rep, ignore_segment, ignore_burning):
        self._seg_len = seg_len
        self._rep = rep
        self._ignore_segment = ignore_segment
        self._ignore_burning = ignore_burning

    def get_task_details(self):
        return {
            "seg_len": self._seg_len,
            "rep": self._rep,
            "ignore_segment": self._ignore_segment,
            "ignore_burning": self._ignore_burning,
        }

    def evaluate_model(
        self,
        model: model_api.Model,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        task_random_seed: Optional[int] = None,
    ):
        if task_random_seed is not None:
            np.random.seed(task_random_seed)
            torch.manual_seed(task_random_seed)

        if batch_size is None:
            if model.model_meta["model_name"].startswith("gpt2"):
                batch_size = 100
            else:
                batch_size = 8

        offset = self._seg_len * self._ignore_segment + self._ignore_burning

        input_ids = make_input_ids(
            num_samples=num_samples,
            seg_len=self._seg_len,
            rep=self._rep,
            vocab_size=model.model_meta["vocab_size"],
            prepend_bos="bos_token_id" in model.model_meta,
            bos=model.model_meta.get("bos_token_id", None),
        )
        input_ids_tensor = torch.Tensor(input_ids).long().to(model.model_meta["device"])

        n_targets = len(input_ids[0]) - offset
        probs_on_correct = np.zeros((num_samples, n_targets))
        pred_next_token_ids = np.zeros((num_samples, n_targets), dtype=np.int64)

        for i, batch in enumerate(torch.split(input_ids_tensor, batch_size)):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_samples)

            batch_logits = model._model(batch).logits
            batch_logits_unmasked = batch_logits[:, offset - 1 : -1]
            batch_probs = torch.nn.functional.softmax(batch_logits_unmasked, dim=-1)
            batch_probs = batch_probs.float().cpu().detach().numpy()

            # Get predictions
            batch_preds = np.argmax(batch_probs, axis=-1)
            pred_next_token_ids[batch_start:batch_end] = batch_preds

            # Get probabilities of correct tokens
            for b in range(len(batch)):
                for s in range(n_targets):
                    probs_on_correct[batch_start + b, s] = batch_probs[
                        b,
                        s,
                        input_ids[
                            batch_start + b,
                            s + offset,
                        ],
                    ]

        errs = np.not_equal(
            input_ids[:, offset:],
            pred_next_token_ids,
        ).astype(float)

        result = result_api.CopyingResult(
            task_details=self.get_task_details(),
            model_details=copy.deepcopy(model.model_meta),
            prob=np.mean(probs_on_correct),
            err=np.mean(errs),
            probs=probs_on_correct.tolist(),
            errs=errs.tolist(),
        )

        return result
