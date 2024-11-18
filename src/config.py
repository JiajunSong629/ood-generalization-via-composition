import transformers
from src.tasks.icl.task import ICLTask
from src.tasks.copying.task import CopyingTask
from src.tasks.gsm.task import GSMTask

MAX_LENGTH = 2048

MODEL_CLASSES = {
    "gpt2": {
        "lm": transformers.GPT2LMHeadModel,
        "tokenizer": transformers.GPT2Tokenizer,
        "hf_name": "gpt2",
    },
    "gpt2-xl": {
        "lm": transformers.GPT2LMHeadModel,
        "tokenizer": transformers.GPT2Tokenizer,
        "hf_name": "gpt2-xl",
    },
    "llama2-7b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "meta-llama/Llama-2-7b-hf",
    },
    "llama3-8b": {
        "lm": transformers.LlamaForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "/home/jiajun/.cache/huggingface/hub/models--meta-llama--Llama-3-8b-hf",
    },
    "gemma-7b": {
        "lm": transformers.GemmaForCausalLM,
        "tokenizer": transformers.GemmaTokenizer,
        "hf_name": "google/gemma-7b",
    },
    "gemma2-9b": {
        "lm": transformers.Gemma2ForCausalLM,
        "tokenizer": transformers.GemmaTokenizer,
        "hf_name": "google/gemma-2-9b",
    },
    "falcon-7b": {
        "lm": transformers.FalconForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "tiiuae/falcon-7b",
    },
    "mistral-7b": {
        "lm": transformers.MistralForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "mistralai/Mistral-7B-v0.1",
    },
    "olmo-7b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "allenai/OLMo-1.7-7B-hf",
    },
    "pythia-7b": {
        "lm": transformers.AutoModelForCausalLM,
        "tokenizer": transformers.AutoTokenizer,
        "hf_name": "EleutherAI/pythia-6.9b",
    },
    # "pythia-14m": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-14m",
    # },
    # "pythia-31m": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-31m",
    # },
    # "pythia-70m": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-70m",
    # },
    # "pythia-160m": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-160m",
    # },
    # "pythia-410m": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-410m",
    # },
    # "pythia-1b": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-1b",
    # },
    # "pythia-1_4b": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-1.4b",
    # },
    # "pythia-2_8b": {
    #     "lm": transformers.AutoModelForCausalLM,
    #     "tokenizer": transformers.AutoTokenizer,
    #     "hf_name": "EleutherAI/pythia-2.8b",
    # },
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
    "llama3-8b": {
        "model_name": "llama3-8b",
        "vocab_size": 32000,
        "num_layers": 32,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "bos_token_id": 1,
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
        "bos_token_id": 0,
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
    # "pythia-14m": {
    #     "model_name": "pythia-14m",
    #     "vocab_size": 50304,
    #     "num_layers": 6,
    #     "num_heads": 4,
    #     "hidden_size": 128,
    #     "head_dim": 32,
    #     "bos_token_id": 0,
    # },
    # "pythia-70m": {
    #     "model_name": "pythia-70m",
    #     "vocab_size": 50304,
    #     "num_layers": 6,
    #     "num_heads": 8,
    #     "hidden_size": 512,
    #     "head_dim": 64,
    #     "bos_token_id": 0,
    # },
    # "pythia-160m": {
    #     "model_name": "pythia-160m",
    #     "vocab_size": 50304,
    #     "num_layers": 12,
    #     "num_heads": 12,
    #     "hidden_size": 768,
    #     "head_dim": 64,
    #     "bos_token_id": 0,
    # },
    # "pythia-410m": {
    #     "model_name": "pythia-410m",
    #     "vocab_size": 50304,
    #     "num_layers": 24,
    #     "num_heads": 16,
    #     "hidden_size": 1024,
    #     "head_dim": 64,
    #     "bos_token_id": 0,
    # },
    # "pythia-1b": {
    #     "model_name": "pythia-1b",
    #     "vocab_size": 50304,
    #     "num_layers": 16,
    #     "num_heads": 8,
    #     "hidden_size": 2048,
    #     "head_dim": 256,
    #     "bos_token_id": 0,
    # },
    # "pythia-1_4b": {
    #     "model_name": "pythia-1.4b",
    #     "vocab_size": 50304,
    #     "num_layers": 24,
    #     "num_heads": 16,
    #     "hidden_size": 2048,
    #     "head_dim": 128,
    #     "bos_token_id": 0,
    # },
    # "pythia-2_8b": {
    #     "model_name": "pythia-2.8b",
    #     "vocab_size": 50304,
    #     "num_layers": 32,
    #     "num_heads": 32,
    #     "hidden_size": 2560,
    #     "head_dim": 80,
    # },
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
    },
    "icl": {
        "task_class": ICLTask,
        "task_kwargs": {
            "setting": "symbol",
            "num_shots": 20,
            "balanced_sample": True,
        },
        "model_kwargs": {"quantize": False},
        "additional_params": ["symbol", "20"],
    },
    "gsm": {
        "task_class": GSMTask,
        "task_kwargs": {
            "num_shots": 10,
            "max_new_tokens": 128,
        },
        "model_kwargs": {"quantize": False},
    },
}
