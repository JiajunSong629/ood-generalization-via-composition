from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import copy
import random
import dataclasses

import src.api.task as task_api
import src.api.result as result_api
import src.api.model as model_api

from src.tasks.fuzzycopy.names import CIFAR_NAMES, VERB_PAIRS


@dataclasses.dataclass
class GenerationExampleResult:
    prompt: str
    expected_answer: str
    model_solution: str
    is_correct: bool
    pred_logprob: float


class FuzzyCopyTask(task_api.Task):
    def __init__(self, setting: str, num_shots: int):
        super().__init__()
        self._setting = setting
        self._num_shots = num_shots

    def get_task_details(self) -> Dict[str, Any]:
        return {
            "name": "FuzzyCopy",
            "description": "Upper/lower case or Past/Current tense copy task.",
            "setting": self._setting,
            "num_shots": self._num_shots,
        }

    def get_examples(
        self,
        num_samples: int,
        task_random_seed: int = None,
    ) -> Tuple[List[str], List[str]]:
        if task_random_seed is not None:
            random.seed(task_random_seed)

        prompts, answers = [], []
        for _ in range(num_samples):
            if self._setting == "upper":
                seq_names = random.choices(CIFAR_NAMES, k=self._num_shots)
                test_names = [name.upper() for name in seq_names]
            elif self._setting == "past":
                pairs = random.choices(VERB_PAIRS, k=self._num_shots)
                seq_names = [pair[0] for pair in pairs]
                test_names = [pair[1] for pair in pairs]
            else:
                raise ValueError(f"Invalid setting: {self._setting}")

            prompt = " ".join(seq_names + test_names[:-1])
            answer = test_names[-1]

            prompts.append(prompt)
            answers.append(answer)

        return prompts, answers

    def evaluate_model(
        self,
        model: model_api.Model,
        num_samples: int | None = None,
        task_random_seed: int = None,
    ):
        prompts, answers = self.get_examples(num_samples, task_random_seed)

        # handle batch size
        if model.model_meta["model_name"].startswith("gpt"):
            batch_size = 100
        else:
            batch_size = 1

        solutions = model.generate_text(
            prompts, max_new_tokens=10, batch_size=batch_size
        )
        scores = model.cond_log_prob(
            inputs=prompts, targets=[[f" {a}"] for a in answers]
        )

        acc = [answer in solution for answer, solution in zip(answers, solutions)]
        eval_results = []
        for prompt, answer, solution, is_correct, score in zip(
            prompts, answers, solutions, acc, scores
        ):
            eval_results.append(
                GenerationExampleResult(
                    prompt=prompt,
                    expected_answer=answer,
                    model_solution=solution,
                    is_correct=is_correct,
                    pred_logprob=score,
                )
            )

        result = result_api.FuzzyCopyResult(
            task_details=self.get_task_details(),
            model_details=copy.deepcopy(model.model_meta),
            acc=float(np.mean(acc)),
            prob=float(np.mean([np.exp(score) for score in scores])),
            num_samples=num_samples,
            random_seed=task_random_seed,
            examples=eval_results,
        )

        return result
