import transformers
import torch
from src.tasks.icl.task import ICLTask
from src.tasks.copying.task import CopyingTask
from src.tasks.gsm.task import GSMTask
# from src.tasks.fuzzycopy.task import FuzzyCopyTask
# from src.tasks.ioi.task import IOITask
from src.tasks.simple_math.task import SimpleMathTask
from src.tasks.gsm_tiny.task import GSMTinyTask

MAX_LENGTH = 2048

MODEL_CLASSES = {
    "gpt2": {
        "lm": transformers.GPT2LMHeadModel,
        "tokenizer": transformers.GPT2Tokenizer,
        "hf_name": "gpt2",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "gpt2-xl": {
        "lm": transformers.GPT2LMHeadModel,
        "tokenizer": transformers.GPT2Tokenizer,
        "hf_name": "gpt2-xl",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "llama2-7b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "llama2-70b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "meta-llama/Llama-2-70b-hf",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "llama3-8b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "llama3-70b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "meta-llama/Meta-Llama-3-70B-Instruct",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "gemma-7b": {
        "lm": transformers.GemmaForCausalLM,
        "tokenizer": transformers.GemmaTokenizer,
        "hf_name": "google/gemma-7b",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "gemma2-9b": {
        "lm": transformers.Gemma2ForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "google/gemma-2-9b",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "falcon-7b": {
        "lm": transformers.FalconForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "tiiuae/falcon-7b",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "mistral-7b": {
        "lm": transformers.MistralForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "olmo-7b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "allenai/OLMo-1.7-7B-hf",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
    "pythia-14m": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-14m",
        "device": "cpu",
        "torch_dtype": torch.float32,
    },
    "pythia-36m": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-31m",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-70m": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-70m",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-160m": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-160m",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-410m": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-410m",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-1b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-1b",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-1_4b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-1.4b",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-2_8b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-2.8b",
        "device": "cuda",
        "torch_dtype": torch.float32,
    },
    "pythia-7b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-6.9b",
        "device": "cuda",
        "torch_dtype": torch.bfloat16,
    },
}

MODEL_META = {
    "gpt2": {
        "model_name": "gpt2",
        "vocab_size": 50257,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "head_dim": 64,
    },
    "gpt2-xl": {
        "model_name": "gpt2-xl",
        "vocab_size": 50257,
        "num_layers": 48,
        "num_heads": 25,
        "hidden_size": 1600,
        "head_dim": 64,
    },
    "llama2-7b": {
        "model_name": "llama2-7b",
        "vocab_size": 32000,
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
        "bos_token_id": 1,
    },
    "llama2-70b": {
        "model_name": "llama2-70b",
        "vocab_size": 32000,
        "num_layers": 80,
        "num_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "head_dim": 128,
        "bos_token_id": 1,
    },
    "llama3-8b": {
        "model_name": "llama3-8b",
        "vocab_size": 128256,
        "num_layers": 32,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "bos_token_id": 128000,
    },
    "llama3-70b": {
        "model_name": "llama3-70b",
        "vocab_size": 128256,
        "num_layers": 80,
        "num_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "head_dim": 128,
        "bos_token_id": 128000,
    },
    "gemma-7b": {
        "model_name": "gemma-7b",
        "vocab_size": 256000,
        "num_layers": 28,
        "num_heads": 16,
        "hidden_size": 3072,
        "head_dim": 256,
        "bos_token_id": 2,
    },
    "gemma2-9b": {
        "model_name": "gemma2-9b",
        "vocab_size": 256000,
        "num_layers": 42,
        "num_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 3584,
        "head_dim": 256,
        "bos_token_id": 2,
    },
    "falcon-7b": {
        "model_name": "falcon-7b",
        "vocab_size": 65024,
        "num_layers": 32,
        "num_heads": 71,
        "hidden_size": 4544,
        "head_dim": 64,
    },
    "mistral-7b": {
        "model_name": "mistral-7b",
        "vocab_size": 32000,
        "num_layers": 32,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "bos_token_id": 1,
    },
    "pythia-7b": {
        "model_name": "pythia-7b",
        "vocab_size": 50304,
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
    },
    "olmo-7b": {
        "model_name": "olmo-7b",
        "vocab_size": 50304,
        "num_layers": 32,
        "num_heads": 32,
        "num_key_value_heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
    },
    "pythia-14m": {
        "model_name": "pythia-14m",
        "vocab_size": 50304,
        "num_layers": 6,
        "num_heads": 4,
        "hidden_size": 128,
        "head_dim": 32,
    },
    "pythia-36m": {
        "model_name": "pythia-36m",
        "vocab_size": 50304,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_size": 256,
        "head_dim": 32,
    },
    "pythia-70m": {
        "model_name": "pythia-70m",
        "vocab_size": 50304,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_size": 512,
        "head_dim": 64,
    },
    "pythia-160m": {
        "model_name": "pythia-160m",
        "vocab_size": 50304,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "head_dim": 64,
    },
    "pythia-410m": {
        "model_name": "pythia-410m",
        "vocab_size": 50304,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_size": 1024,
        "head_dim": 64,
    },
    "pythia-1b": {
        "model_name": "pythia-1b",
        "vocab_size": 50304,
        "num_layers": 16,
        "num_heads": 8,
        "hidden_size": 2048,
        "head_dim": 256,
    },
    "pythia-1_4b": {
        "model_name": "pythia-1.4b",
        "vocab_size": 50304,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_size": 2048,
        "head_dim": 128,
    },
    "pythia-2_8b": {
        "model_name": "pythia-2.8b",
        "vocab_size": 50304,
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 2560,
        "head_dim": 80,
    },
}


TASK_CONFIGS = {
    "copying": {
        "task_class": CopyingTask,
        "task_kwargs": {
            "seg_len": 25,
            "rep": 3,
            "ignore_segment": 2,
            "ignore_burning": 4,
        },
        "model_kwargs": {"quantize": False},
        "eval_kwargs": {
            "num_samples": 50,
            "task_random_seed": 2024,
        },
    },
    "icl": {
        "task_class": ICLTask,
        "task_kwargs": {
            "setting": "symbol",
            "num_shots": 20,
            "num_beams": 5,
            "num_outputs": 3,
            "balanced_sample": True,
        },
        "model_kwargs": {"quantize": False},
        "eval_kwargs": {
            "num_samples": 100,
            "task_random_seed": 1234,  # zhuoyan's version, seed is 1234
        },
        "additional_params": ["symbol", "20"],
    },
    "icl-original": {
        "task_class": ICLTask,
        "task_kwargs": {
            "setting": "original",
            "num_shots": 20,
            "num_beams": 5,
            "num_outputs": 3,
            "balanced_sample": True,
        },
        "model_kwargs": {"quantize": False},
        "eval_kwargs": {
            "num_samples": 100,
            "task_random_seed": 1234,  # zhuoyan's version, seed is 1234
        },
        "additional_params": ["original", "20"],
    },
    "gsm": {
        "task_class": GSMTask,
        "task_kwargs": {
            "num_shots": 10,
            "max_new_tokens": 128,
            "num_beams": 1,
            "num_outputs": 1,
        },
        "eval_kwargs": {
            "num_samples": 100,
            "task_random_seed": 42,
        },
        "model_kwargs": {"quantize": False},
        "additional_params": ["10"],
    },
    # "fuzzycopy": {
    #     "task_class": FuzzyCopyTask,
    #     "task_kwargs": {
    #         "setting": "upper",
    #         "num_shots": 8,
    #         "num_tests": 2,
    #     },
    #     "eval_kwargs": {
    #         "num_samples": 100,
    #         "task_random_seed": 1234,
    #     },
    #     "model_kwargs": {"quantize": False},
    #     "additional_params": ["upper", "8"],
    # },
    # "ioi": {
    #     "task_class": IOITask,
    #     "task_kwargs": {
    #         "setting": "symbol",
    #         "num_beams": 5,
    #         "num_outputs": 3,
    #         "max_new_tokens": 5,
    #     },
    #     "eval_kwargs": {
    #         "num_samples": 100,
    #         "task_random_seed": 42,  # in zhuoyan's version, seed is 42
    #     },
    #     "model_kwargs": {"quantize": False},
    #     "additional_params": ["symbol"],
    # },
    # "ioi-original": {
    #     "task_class": IOITask,
    #     "task_kwargs": {
    #         "setting": "original",
    #         "num_beams": 5,
    #         "num_outputs": 3,
    #         "max_new_tokens": 5,
    #     },
    #     "eval_kwargs": {
    #         "num_samples": 100,
    #         "task_random_seed": 42,  # in zhuoyan's version, seed is 42
    #     },
    #     "model_kwargs": {"quantize": False},
    #     "additional_params": ["original"],
    # },
    "simple_math": {
        "task_class": SimpleMathTask,
        "task_kwargs": {
            "num_shots": 4,
            "setting": "symbol",
            "max_new_tokens": 80,
        },
        "eval_kwargs": {
            "num_samples": 100,
            "task_random_seed": 42,
        },
        "model_kwargs": {"quantize": False},
        "additional_params": ["symbol", "4"],
    },
    "gsm_tiny": {
        "task_class": GSMTinyTask,
        "task_kwargs": {
            "num_shots": 10,
            "setting": "symbol",
            "max_new_tokens": 256,
        },
        "eval_kwargs": {
            "num_samples": 100,
            "task_random_seed": 42,
        },
        "model_kwargs": {"quantize": False},
        "additional_params": ["10"],
    },
}


PROJECT_CONFIGS = {
    "copying": {
        "project_n_heads": 10,
        "projected_n_heads": 0.25,
    },
    "ioi": {
        "project_n_heads": 10,
        "projected_n_heads": 50,
        "additional_params": ["50"],
    },
    "icl": {
        "project_n_heads": 10,
        "projected_n_heads": 50,
        "additional_params": ["50"],
    },
    "fuzzycopy": {
        "project_n_heads": 10,
        "projected_n_heads": 50,
        "additional_params": ["50"],
    },
    "gpt2": {"ranks": [0, 50, 60]},
    "gpt2-xl": {"ranks": [0, 80, 90]},
    "llama2-7b": {"ranks": [0, 50, 100, 200, 210]},
    "llama3-8b": {"ranks": [0, 50, 100, 200, 210]},
    "gemma-7b": {"ranks": [0, 50, 100, 150, 160]},
    "gemma2-9b": {"ranks": [0, 50, 100, 170, 180]},
    "falcon-7b": {"ranks": [0, 50, 100, 200, 220, 230]},
    "mistral-7b": {"ranks": [0, 50, 100, 200, 210]},
    "olmo-7b": {"ranks": [0, 50, 100, 200, 210]},
    "pythia-7b": {"ranks": [0, 50, 100, 200, 210]},
}

SHUFFLE_CONFIGS = {
    "shuffle_seeds": range(0, 100, 20),
    "shuffle_n_heads": 10,
}

REMOVAL_CONFIGS = {
    "random_seeds": [37, 42, 134, 1567, 8787],  # range(0, 100, 20),
    # "gpt2": {"n_heads": range(0, 60, 10)},
    # "gpt2-xl": {"n_heads": range(0, 60, 10)},
    # "llama2-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "llama3-8b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "gemma-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "gemma2-9b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "falcon-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "mistral-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    # "olmo-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
    "gpt2": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "gpt2-xl": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "llama2-7b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "llama3-8b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "llama2-70b": {"n_heads": [0, 50, 100, 150, 200, 250]},
    "llama3-70b": {"n_heads": [0, 50, 100, 150, 200, 250]},
    "gemma-7b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "gemma2-9b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "falcon-7b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "mistral-7b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    "olmo-7b": {"n_heads": [0, 10, 20, 30, 40, 50]},
    # pythia models
    "pythia-14m": {"n_heads": range(0, 40, 10)},
    "pythia-36m": {"n_heads": range(0, 60, 10)},
    "pythia-70m": {"n_heads": range(0, 60, 10)},
    "pythia-160m": {"n_heads": range(0, 60, 10)},
    "pythia-410m": {"n_heads": range(0, 60, 10)},
    "pythia-1b": {"n_heads": range(0, 60, 10)},
    "pythia-1_4b": {"n_heads": range(0, 60, 10)},
    "pythia-2_8b": {"n_heads": range(0, 60, 10)},
    "pythia-7b": {"n_heads": list(range(0, 60, 10)) + [100, 200]},
}
