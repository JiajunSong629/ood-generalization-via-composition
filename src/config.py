import transformers

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
        "num_heads": 32,
        "hidden_size": 4096,
        "head_dim": 128,
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
        "vocab_size": 50432,
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
}


# ATTENTION_LAYER_PATHS = {
#     "gpt2": lambda model: model.transformer.h,  # Returns list of transformer blocks
#     "gpt2-xl": lambda model: model.transformer.h,
#     "llama2-7b": lambda model: model.model.layers,
#     "gemma-7b": lambda model: model.model.layers,
#     "falcon-7b": lambda model: model.transformer.h,
#     "mistral-7b": lambda model: model.model.layers,
#     "olmo-7b": lambda model: model.model.layers,
#     "pythia-7b": lambda model: model.gpt_neox.layers,
#     "llama2-70b": lambda model: model.model.layers,
#     "gemma-9b": lambda model: model.model.layers,
# }

# ATTENTION_MODULE_NAMES = {
#     "gpt2": "attn",
#     "gpt2-xl": "attn",
#     "llama2-7b": "self_attn",
#     "gemma-7b": "self_attn",
#     "falcon-7b": "self_attention",
#     "mistral-7b": "self_attn",
#     "olmo-7b": "self_attn",
#     "pythia-7b": "attention",
#     "llama2-70b": "self_attn",
#     "gemma-9b": "self_attn",
# }
