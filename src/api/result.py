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
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return f"[{self.model_details['model_name']} | {time_str}] CopyingResult(acc={1 - self.err:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class ICLResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float
    prob: float

    def __repr__(self) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return f"[{self.model_details['model_name']} | {time_str}] ICLResult(acc={self.acc:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class FuzzyCopyResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float
    prob: float

    def __repr__(self) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return f"[{self.model_details['model_name']} | {time_str}] FuzzyCopyResult(acc={self.acc:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class IOIResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float
    prob: float

    def __repr__(self) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return f"[{self.model_details['model_name']} | {time_str}] IOIResult(acc={self.acc:.2f}, prob={self.prob:.2f})"


@dataclasses.dataclass(kw_only=True)
class GSMResult(BaseResult):
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    acc: float

    def __repr__(self) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return f"[{self.model_details['model_name']} | {time_str}] GSMResult(acc={self.acc:.2f})"
