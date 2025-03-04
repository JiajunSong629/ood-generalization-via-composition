import random
import dataclasses
from typing import List, Optional, Union
import copy
import re

import src.api.task as task
import src.api.model as model
import src.api.result as result_api

from src.tasks.gsm_rand.template import generate_task_with_context


@dataclasses.dataclass
class GenerationExampleResult:
    prompt: str
    expected_answer: str
    model_solution: str
    model_answer: str
    correct: bool


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class GSMRandTask(task.Task):
    def __init__(
        self,
        num_shots: Optional[int] = 3,
        num_beams: Optional[int] = 1,
        num_outputs: Optional[int] = 1,
        max_new_tokens: Optional[int] = 80,
        setting: str = "symbol",
    ):
        super().__init__()
        self._num_shots = num_shots
        self._num_beams = num_beams
        self._num_outputs = num_outputs
        self._max_new_tokens = max_new_tokens
        self._setting = setting

    def get_task_details(self) -> task.TaskMetadata:
        return {
            "name": "gsm_rand",
            "description": "GSM word problems with randomizations on names or numbers",
            "num_shots": self._num_shots,
            "num_outputs": self._num_outputs,
            "max_new_tokens": self._max_new_tokens,
        }

    def _extract_answer(self, completion: str) -> str:
        completion = completion.replace("$", "")
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def _generate_examples(self, num_samples: int):
        prompts, answers = [], []
        for _ in range(num_samples):
            example = generate_task_with_context(
                self._setting, num_shots=self._num_shots
            )

            prompts.append(example["prompt"])
            answers.append(example["answer"])

        return prompts, answers

    def evaluate_model(
        self,
        model: model.Model,
        num_samples: Optional[int] = None,
        task_random_seed: Optional[int] = 0,
    ) -> Union[task.ScoreData, List[task.ScoreData]]:
        if task_random_seed is not None:
            random.seed(task_random_seed)

        prompts, answers = self._generate_examples(num_samples)
        acc = 0
        examples = []

        for prompt, expected_answer in zip(prompts, answers):
            response = model.generate_text(
                prompt,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                num_outputs=self._num_outputs,
                batch_size=1,
            )

            if isinstance(response, list):
                model_answer = [self._extract_answer(r) for r in response]
                cur_result = expected_answer in model_answer
            else:
                model_answer = self._extract_answer(response)
                cur_result = model_answer == expected_answer

            examples.append(
                GenerationExampleResult(
                    prompt=prompt,
                    expected_answer=expected_answer,
                    model_solution=response,
                    model_answer=model_answer,
                    correct=cur_result,
                )
            )
            acc += cur_result

        result = result_api.GSMRandResult(
            model_details=copy.deepcopy(model.model_meta),
            task_details=self.get_task_details(),
            acc=acc / num_samples,
            num_samples=num_samples,
            random_seed=task_random_seed,
            examples=examples,
        )

        return result
