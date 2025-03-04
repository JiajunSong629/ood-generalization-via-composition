from typing import List, Optional, Dict, Any
import dataclasses
import json
import src.api.task as task_api
import src.api.model as model_api
import time


@dataclasses.dataclass(kw_only=True)
class BaseResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def save_multiple(cls, results: List["BaseResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


@dataclasses.dataclass(kw_only=True)
class CopyingResult(BaseResult):
    prob: float
    err: float
    probs: List[float]
    errs: List[float]

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] CopyingResult(acc={1 - self.err:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class ICLResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc_generation: float
    acc_multiple_choice: float
    prob: float

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] ICLResult(acc_generation={self.acc_generation:.2f}, acc_multiple_choice={self.acc_multiple_choice:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class FuzzyCopyResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float
    prob: float

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] FuzzyCopyResult(acc={self.acc:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class IOIResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc_generation: float
    acc_multiple_choice: float
    prob: float

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] IOIResult(acc_generation={self.acc_generation:.2f}, acc_multiple_choice={self.acc_multiple_choice:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class GSMResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] GSMResult(acc={self.acc:.2f})"


@dataclasses.dataclass(kw_only=True)
class SimpleMathResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float

    def __repr__(self) -> str:
        return (
            f"[{self.model_details['model_name']}] SimpleMathResult(acc={self.acc:.2f})"
        )


@dataclasses.dataclass(kw_only=True)
class GSMRandResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float

    def __repr__(self) -> str:
        return f"[{self.model_details['model_name']}] GSMRandResult(acc={self.acc:.2f})"
