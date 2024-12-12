from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
import copy

import src.api.task as task_api
import src.api.result as result_api
import src.api.model as model_api

torch.set_grad_enabled(False)


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

    def make_input_ids(
        self,
        num_samples,
        vocab_size,
        prepend_bos,
        bos=None,
        task_random_seed=None,
    ) -> np.ndarray:
        if task_random_seed is not None:
            np.random.seed(task_random_seed)

        sample_int = np.random.randint(
            low=0,
            high=vocab_size,
            size=num_samples * self._seg_len,
        ).reshape(num_samples, self._seg_len)
        sample_int = np.concatenate(tuple([sample_int] * self._rep), axis=1)

        if prepend_bos:
            assert bos is not None
            sample_int = np.hstack(
                [bos * np.ones((num_samples, 1), dtype=int), sample_int]
            )
        input_ids = np.array(sample_int, dtype=np.int32)

        return input_ids

    def evaluate_model(
        self,
        model: model_api.Model,
        num_samples: Optional[int] = None,
        task_random_seed: Optional[int] = None,
    ):
        input_ids = self.make_input_ids(
            num_samples=num_samples,
            vocab_size=model.model_meta["vocab_size"],
            prepend_bos="bos_token_id" in model.model_meta,
            bos=model.model_meta.get("bos_token_id", None),
            task_random_seed=task_random_seed,
        )
        input_ids_tensor = torch.Tensor(input_ids).long().to(model.model_meta["device"])

        with torch.no_grad():
            for i in range(input_ids_tensor.size(0)):
                cur_batch_input_ids = input_ids_tensor[i : i + 1]
                cur_logits = model._model(cur_batch_input_ids).logits.cpu()
                if i == 0:
                    logits = cur_logits
                else:
                    logits = torch.concat([logits, cur_logits])

        probs = F.softmax(logits.float(), dim=-1)
        _, pred_next_token_ids = torch.topk(probs, dim=-1, k=1)

        errs = (input_ids_tensor[:, 1:].cpu() != pred_next_token_ids[:, :-1, 0]).numpy(
            force=True
        )

        probs_on_correct = np.zeros_like(input_ids_tensor[:, 1:].numpy(force=True))
        _, seq_len, n_vocab = probs.shape
        probs_on_correct = np.zeros((num_samples, seq_len - 1))
        for b in range(num_samples):
            for s in range(seq_len - 1):
                probs_on_correct[b, s] = probs[b, s, input_ids_tensor[b, s + 1]]

        T_range = range(
            self._seg_len * self._ignore_segment + self._ignore_burning - 1,
            self._rep * self._seg_len - 1,
        )
        probs_on_correct = probs_on_correct[:, T_range]
        errs = errs[:, T_range]

        result = result_api.CopyingResult(
            task_details=self.get_task_details(),
            model_details=copy.deepcopy(model.model_meta),
            prob=np.mean(probs_on_correct),
            err=np.mean(errs),
            probs=probs_on_correct.tolist(),
            errs=errs.tolist(),
        )

        return result
