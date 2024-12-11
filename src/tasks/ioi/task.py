from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import copy
import random
import dataclasses

import src.api.task as task_api
import src.api.result as result_api
import src.api.model as model_api

from src.tasks.ioi.names import NAMES, OBJECTS, BABA_TEMPLATES, PLACES, SYMBOLS


@dataclasses.dataclass
class GenerationExampleResult:
    prompt: str
    expected_answer: str
    generation_solution: str
    multiple_choice_solution: str
    is_correct_generation: bool
    is_correct_multiple_choice: bool
    prob: float
    multiple_choice_logprob: Dict[str, float]


class IOITask(task_api.Task):
    def __init__(
        self,
        setting: str,
        num_beams: int = 1,
        num_outputs: int = 1,
        max_new_tokens: int = 5,
    ):
        self._setting = setting
        self._num_beams = num_beams
        self._num_outputs = num_outputs
        self._max_new_tokens = max_new_tokens
        super().__init__()

    def get_task_details(self) -> Dict[str, Any]:
        return {
            "name": "ioi",
            "description": "IOI tasks",
            "num_beams": self._num_beams,
            "num_outputs": self._num_outputs,
            "max_new_tokens": self._max_new_tokens,
        }

    def get_examples(
        self, num_examples: int, task_random_seed: int = None
    ) -> List[List[str]]:
        if task_random_seed is not None:
            random.seed(1234)  # hard coding seed 1234 to reproduce zhuoyan's results
            np.random.seed(task_random_seed)

        prompts, answers, choices = [], [], []
        for _ in range(num_examples):
            if self._setting == "symbol":
                name_A, name_B = np.random.choice(SYMBOLS, 2, replace=False)
            else:
                name_A, name_B = np.random.choice(NAMES, 2, replace=False)

            place = random.choice(PLACES)
            obj = random.choice(OBJECTS)

            prompt = random.choice(BABA_TEMPLATES)
            prompt = prompt.replace("[A]", name_A)
            prompt = prompt.replace("[B]", name_B)
            prompt = prompt.replace("[PLACE]", place)
            prompt = prompt.replace("[OBJECT]", obj)

            prompt, answer = prompt[: -len(name_A) - 1], f"{name_A}"
            choice = [f" {name_A}", f" {name_B}"]

            prompts.append(prompt)
            answers.append(answer)
            choices.append(choice)

        return prompts, answers, choices

    def evaluate_model(
        self,
        model: model_api.Model,
        num_samples: int | None = None,
        task_random_seed: int | None = 0,
    ):
        prompts, answers, choices = self.get_examples(num_samples, task_random_seed)
        if model.model_meta["model_name"].startswith("gpt"):
            batch_size = 8
        else:
            batch_size = 1

        solutions = model.generate_text(
            prompts,
            max_new_tokens=self._max_new_tokens,
            batch_size=batch_size,
            num_beams=self._num_beams,
            num_outputs=self._num_outputs,
        )
        scores = model.cond_log_prob(inputs=prompts, targets=choices)

        eval_results = []
        for prompt, answer, score, generation_solution, choice in zip(
            prompts, answers, scores, solutions, choices
        ):
            logprob_on_correct = score[choice.index(f" {answer}")]
            is_correct_multiple_choice = np.argmax(score) == choice.index(f" {answer}")
            multiple_choice_solution = choice[np.argmax(score)]

            if isinstance(generation_solution, list):
                is_correct_generation = any(answer in s for s in generation_solution)
            else:
                is_correct_generation = answer in generation_solution

            eval_results.append(
                GenerationExampleResult(
                    prompt=prompt,
                    expected_answer=answer,
                    generation_solution=generation_solution,
                    multiple_choice_solution=multiple_choice_solution,
                    is_correct_generation=float(is_correct_generation),
                    is_correct_multiple_choice=float(is_correct_multiple_choice),
                    prob=np.exp(logprob_on_correct),
                    multiple_choice_logprob={c: lp for c, lp in zip(choice, score)},
                )
            )

        result = result_api.IOIResult(
            task_details=self.get_task_details(),
            model_details=copy.deepcopy(model.model_meta),
            acc_generation=float(
                np.mean([r.is_correct_generation for r in eval_results])
            ),
            acc_multiple_choice=float(
                np.mean([r.is_correct_multiple_choice for r in eval_results])
            ),
            prob=float(np.mean([r.prob for r in eval_results])),
            num_samples=num_samples,
            random_seed=task_random_seed,
            examples=eval_results,
        )

        return result
