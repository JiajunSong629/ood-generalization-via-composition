from dataclasses import dataclass, asdict, field
from typing import List
from typing import Callable
from pathlib import Path
import torch
import random
import os
import torch
import torch.nn as nn
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from pdb import set_trace as pds

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# from .llama.get_id import get_top_layer_head_ids_llama
from .llama import ModelArgs, Transformer, Tokenizer
from .llama import Ada_LlamaForCausalLM, llama2_layer_head_id_7b, llama2_layer_head_id_70b, llama3_layer_head_id_8b
from transformers import LlamaTokenizer, AutoTokenizer
from .mistral import Ada_MistralForCausalLM, mistral_layer_head_id
from .gemma import Ada_GemmaForCausalLM, gemma_layer_head_id_7b, gemma2_layer_head_id_9b
from .falcon import Ada_FalconForCausalLM, falcon_layer_head_id_7b, falcon_layer_head_id_11b
from .olmo import Ada_OlmoForCausalLM, olmo_layer_head_id_7b
from .pythia import Ada_PythiaForCausalLM, pythia_layer_head_id_7b
from .utils import Timer, time_str

import pdb

@dataclass
class EvalResult:
    prompt: str
    solution: str
    answer: str
    accuracy: bool
    topk_solutions: List[str] = field(default_factory=list)  # Default empty list
    topk_probs: List[str] = field(default_factory=list)  # Default empty list
    topk_loss: List[str] = field(default_factory=list)  # Default empty list
    loss_diff: float = 0.0
    topk_accuracy: bool = False  # Default value


class Evaluator:
    def __init__(self, prompt_to_solution: Callable) -> None:
        self.prompt_to_solution = prompt_to_solution

    def eval(self, prompt, answer) -> EvalResult:
        best_solution, topk_solutions, topk_probs = self.prompt_to_solution(prompt)

        topk_accuracy = any(answer in solution for solution in topk_solutions)

        result = EvalResult(
            prompt=prompt, 
            solution=best_solution,
            answer=answer,
            accuracy=answer in best_solution,
            topk_solutions=topk_solutions,
            topk_probs=topk_probs,
            topk_accuracy=topk_accuracy
        )
        return result
    
    def eval_gsm(self, prompt, answer) -> EvalResult:
        solution = self.prompt_to_solution(prompt)

        result = {
            "prompt": prompt, 
            "solution": solution,
            "answer": answer,
        }

        # result = EvalResult(
        #     prompt=prompt, 
        #     solution=solution,
        #     answer=answer,
        #     accuracy=answer in solution,
        # )
        return result
    
    def eval_loss(self, prompt, truth, choices: List = []) -> EvalResult:
        losses = []
        # Find the choice with the smallest loss
        min_loss = float('inf')
        best_choice = None
        for answer in choices:
            loss = self.prompt_to_solution(prompt, answer)
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                best_choice = answer
        loss_diff = losses[0] - losses[1]


        result = EvalResult(
            prompt=prompt, 
            solution=best_choice,
            answer=truth,
            accuracy=truth in best_choice,
            topk_solutions=choices,
            topk_loss=losses,
            loss_diff = loss_diff
        )
        return result

def load_model(model_path = "meta-llama/Llama-2-7b-hf", mask_head = 0, random_mask_head = False, random_mask_head_seed = 42):
    if random_mask_head:
        # set local random seed for select heads
        rnd = random.Random()
        rnd.seed(random_mask_head_seed)
        print(f"random_mask_head_seed = {random_mask_head_seed}")
    print(f"Loading {model_path} from huggingface...")
    if "Llama-2" in model_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

        if "7b" in model_path:
            ### notes: need to specify eager here, then can use customized attention module!
            # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=False, device_map="auto")
            model = Ada_LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=False, device_map="auto",attn_implementation = "eager" )
            # attn_implementation = "eager"
            # print("implementation: ", model.config._attn_implementation)
            mask_layer_head_id = llama2_layer_head_id_7b
            all_pairs = [(x, y) for x in range(32) for y in range(32)]
            if mask_head > 0:
                if random_mask_head:
                    # Randomly subsample mask_head indices without replacement
                    mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                    # print("random sample mask_layer_head_id: ", mask_layer_head_id)
                else:
                    mask_layer_head_id = mask_layer_head_id[:mask_head]
                    # print("slice mask_layer_head_id: ", mask_layer_head_id)
                model.set_mask_layer_head_id(mask_layer_head_id)

        elif "70b" in model_path:
            timer = Timer()
            model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=False, low_cpu_mem_usage=True, device_map="auto", attn_implementation = "eager")
            print(f"load 70b model done, time {time_str(timer.end())}")

            '''
            model = Ada_LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=False, low_cpu_mem_usage=True, device_map="auto", attn_implementation = "eager")

            max_memory = get_balanced_memory(
                model,
                max_memory=None,
                no_split_module_classes=["Ada_LlamaMLP"],
                dtype='float16',
                low_zero=False,
            )

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["Ada_LlamaMLP"],
                dtype='float16'
            )
            model = dispatch_model(model, device_map=device_map)
            #
            '''
            # mask_layer_head_id = get_top_layer_head_ids_llama(scale = "70b", num_heads = 300)
            mask_layer_head_id = llama2_layer_head_id_70b
            all_pairs = [(x, y) for x in range(80) for y in range(64)]

            if mask_head > 0:
                print("have mask head, load modified weights")
                ckpt_file = f"llama_2_70b_mask{mask_head}.pth"
                if random_mask_head:
                    # Load your modified state_dict
                    ckpt_file = f"llama_2_70b_mask{mask_head}_random_seed{random_mask_head_seed}.pth"

                modified_ckpt_path = os.path.join(current_dir, "llama","ckpt",ckpt_file)
                print(f"torch load start, time {time_str(timer.end())}")
                modified_ckpt = torch.load(modified_ckpt_path, map_location=torch.device('cpu'))
                print(f"torch load done, time {time_str(timer.end())}")
                # Apply the modified state_dict to your model
                model.load_state_dict(modified_ckpt)
                print(f"load state dict done, time {time_str(timer.end())}")
        else:
            raise NotImplementedError("not implemented yet")
    
    elif "Llama-3" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if "8B" in model_path:
            ### notes: need to specify eager here, then can use customized attention module!
            # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=False, device_map="auto")
            model = Ada_LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=False, device_map="auto",attn_implementation = "eager" )
            # attn_implementation = "eager"
            # print("implementation: ", model.config._attn_implementation)
            mask_layer_head_id = llama3_layer_head_id_8b
            all_pairs = [(x, y) for x in range(32) for y in range(32)]
            if mask_head > 0:
                if random_mask_head:
                    # Randomly subsample mask_head indices without replacement
                    mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                    # print("random sample mask_layer_head_id: ", mask_layer_head_id)
                else:
                    mask_layer_head_id = mask_layer_head_id[:mask_head]
                    # print("slice mask_layer_head_id: ", mask_layer_head_id)
                model.set_mask_layer_head_id(mask_layer_head_id)

    elif "mistralai" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Ada_MistralForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=True)
        mask_layer_head_id = mistral_layer_head_id
        all_pairs = [(x, y) for x in range(32) for y in range(32)]
        if mask_head > 0:
            if random_mask_head:
                # Randomly subsample mask_head indices without replacement
                mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                # print("random sample mask_layer_head_id: ", mask_layer_head_id)
            else:
                mask_layer_head_id = mask_layer_head_id[:mask_head]
                # print("slice mask_layer_head_id: ", mask_layer_head_id)
            model.set_mask_layer_head_id(mask_layer_head_id)
    
        model = model.to(device)
    
    elif "gemma" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Ada_GemmaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True, attn_implementation = "eager")
        if "gemma-7b-it" in model_path:
            mask_layer_head_id = gemma_layer_head_id_7b
            all_pairs = [(x, y) for x in range(28) for y in range(16)]
        elif "gemma-7b" in model_path:
            mask_layer_head_id = gemma_layer_head_id_7b
            all_pairs = [(x, y) for x in range(28) for y in range(16)]
        elif "gemma-2-9b" in model_path:
            mask_layer_head_id = gemma2_layer_head_id_9b
            all_pairs = [(x, y) for x in range(42) for y in range(16)]
        if mask_head > 0:
            if random_mask_head:
                # Randomly subsample mask_head indices without replacement
                mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                print("random sample mask_layer_head_id: ", mask_layer_head_id)
            else:
                mask_layer_head_id = mask_layer_head_id[:mask_head]
                print("slice mask_layer_head_id: ", mask_layer_head_id)
            model.set_mask_layer_head_id(mask_layer_head_id)
    
        model = model.to(device)
    
    elif "falcon" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Ada_FalconForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True, attn_implementation = "eager")
        if "7b" in model_path:
            mask_layer_head_id = falcon_layer_head_id_7b
            all_pairs = [(x, y) for x in range(32) for y in range(71)]
        elif "11B" in model_path:
            mask_layer_head_id = falcon_layer_head_id_11b
            all_pairs = [(x, y) for x in range(60) for y in range(32)]
        if mask_head > 0:
            if random_mask_head:
                # Randomly subsample mask_head indices without replacement
                mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                print("random sample mask_layer_head_id: ", mask_layer_head_id)
            else:
                mask_layer_head_id = mask_layer_head_id[:mask_head]
                print("slice mask_layer_head_id: ", mask_layer_head_id)
            model.set_mask_layer_head_id(mask_layer_head_id)
        model = model.to(device)
    
    elif "OLMo" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Ada_OlmoForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True, attn_implementation = "eager")
        mask_layer_head_id = olmo_layer_head_id_7b
        all_pairs = [(x, y) for x in range(32) for y in range(32)]
        if mask_head > 0:
            if random_mask_head:
                # Randomly subsample mask_head indices without replacement
                mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                print("random sample mask_layer_head_id: ", mask_layer_head_id)
            else:
                mask_layer_head_id = mask_layer_head_id[:mask_head]
                print("slice mask_layer_head_id: ", mask_layer_head_id)
            model.set_mask_layer_head_id(mask_layer_head_id)
    
        model = model.to(device)
    
    elif "pythia" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Ada_PythiaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_attentions=True, attn_implementation = "eager")
        
        mask_layer_head_id = pythia_layer_head_id_7b
        all_pairs = [(x, y) for x in range(32) for y in range(32)]
        if mask_head > 0:
            if random_mask_head:
                # Randomly subsample mask_head indices without replacement
                mask_layer_head_id = rnd.sample(all_pairs, mask_head)
                print("random sample mask_layer_head_id: ", mask_layer_head_id)
            else:
                mask_layer_head_id = mask_layer_head_id[:mask_head]
                print("slice mask_layer_head_id: ", mask_layer_head_id)
            model.set_mask_layer_head_id(mask_layer_head_id)
    
        model = model.to(device)

    model.eval()
    print("load model Done.")

    return model, tokenizer

@torch.no_grad()
def make_prompt_to_solution(model_path = "meta-llama/Llama-2-7b-hf", mask_head = 0, random_mask_head = False, beam_width: int = 5, seq_len = 20, random_mask_head_seed = 42):
    model, tokenizer = load_model(model_path, mask_head, random_mask_head, random_mask_head_seed = random_mask_head_seed)

    def prompt_to_solution(prompt, seq_len: int = seq_len, beam_width = beam_width):
        # Assume tokenizer and model are defined and configured for CUDA as in the original function.
        prompt_len = len(prompt) 
        # Encode the input prompt
        tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        # print("prompt tok: ", tokens)
        # This will store the candidate sequences and their log probabilities (initialize with the prompt)
        candidates = [(tokens, 0)]  # List of tuples (token_tensor, log_prob)
        
        for _ in range(seq_len):
            new_candidates = []
            for candidate, log_prob in candidates:
                # Use the model's forward pass for each candidate
                outputs = model(input_ids=candidate, return_dict=True)
                logits = outputs.logits[:, -1, :]  # Focus on the logits for the last token position
                
                # Apply softmax to convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                # Get the top beam_width indices and their log probabilities
                top_probs, top_indices = torch.topk(probs, beam_width)
                
                # print("top probs: ", top_probs, top_probs.shape)
                # Iterate through the top tokens and update the candidates list
                for i in range(beam_width):
                    new_token = top_indices[0][i].unsqueeze(0).unsqueeze(-1)
                    # print(new_token, new_token.shape)
                    #assert False
                    new_log_prob = log_prob + torch.log(top_probs[0][i])
                    new_candidate = torch.cat((candidate, new_token), dim=1)
                    
                    new_candidates.append((new_candidate, new_log_prob))
                
            ## Select the top beam_width candidates based on their log probabilities
            new_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by log probability
            candidates = new_candidates[:beam_width]
            
        # Choose the best candidate after the final iteration
        best_candidate, _ = candidates[0]
        # Decode the token IDs to a string
        solution = tokenizer.decode(best_candidate.squeeze().tolist(), skip_special_tokens=True)
        solution = solution[prompt_len:]

        # Initialize a list to store the top-5 solutions
        solutions = []
        probs = []
        # Iterate through the top-5 candidates and decode each one
        for candidate, prob in candidates:
            # Decode the token IDs to a string
            decoded_solution = tokenizer.decode(candidate.squeeze().tolist(), skip_special_tokens=True)
            solutions.append(decoded_solution[prompt_len:])
            probs.append(torch.exp(prob).item())

        return solution, solutions, probs
    
    return prompt_to_solution


@torch.no_grad()
def generate_prompt_to_solution(model_path = "meta-llama/Llama-2-70b-hf", mask_head = 0, random_mask_head = False, seq_len = 20, random_mask_head_seed = 42):
    '''
    similar to make_promt_ro_solution, however, in this function, we do not manually autoregressive inputs to model, we use model.generate function to achieve the goal
    '''

    print("generate prompt to solution func random mask head seed: ", random_mask_head_seed)
    model, tokenizer = load_model(model_path, mask_head, random_mask_head, random_mask_head_seed = random_mask_head_seed)

    def prompt_to_solution(prompt, seq_len: int = seq_len):
        # Assume tokenizer and model are defined and configured for CUDA as in the original function.
        prompt_len = len(prompt) 
        # Encode the input prompt
        tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **tokens, max_new_tokens=256, pad_token_id=model.config.eos_token_id
        )
        solution = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        solution = solution[prompt_len:]
        
        return solution
    
    return prompt_to_solution


@torch.no_grad()
def make_prompt_to_loss(model_path = "meta-llama/Llama-2-7b-hf", mask_head = 0, random_mask_head = False, random_mask_head_seed = 42):
    print("make prompt to loss func random mask head seed: ", random_mask_head_seed)
    model, tokenizer = load_model(model_path, mask_head, random_mask_head, random_mask_head_seed = random_mask_head_seed)

    def prompt_to_loss(prompt, answer):
        # print(f"prompt: =={prompt}==")
        # print(f"answer: =={answer}==")
        # print(f"+: =={prompt} {answer}==")

        # Encode the input prompt and answer
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        answer_tokens = tokenizer.encode(answer, return_tensors="pt").to("cuda")
        combine_tokens = tokenizer.encode(f"{prompt} {answer}", return_tensors="pt").to("cuda")

        # print("prompt tok: ", prompt_tokens.shape, prompt_tokens)
        # print("ans tok: ",answer_tokens.shape,  answer_tokens)
        # print("combine tok: ", combine_tokens.shape, combine_tokens)
        # print("decode combine_tok: ", tokenizer.decode(combine_tokens.tolist()[0]))

        # Concatenate prompt and answer tokens # TODO, should not concat tokens together, should do A then A+B
        concat_tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)

        # print("concat tok: ", concat_tokens.shape, concat_tokens)
        # print(concat_tokens.tolist()[0])
        # print("decode concat_tok: ", tokenizer.decode(concat_tokens.tolist()[0]))

        # Get the length of the prompt and answer
        prompt_len = prompt_tokens.size(1)
        # print("prompt_len: ", prompt_len)
        
        # Forward pass through the model
        outputs = model(input_ids=combine_tokens, return_dict=True)
        # print("output.logits: ", outputs.logits.shape)
        logits = outputs.logits[:, prompt_len-1:-1, :]  # tricky, depends on the tokenizer, sos and eos token, better to print out then trunctate
        ### logits = outputs.logits[:, :-1, :]  # tricky, depends on the tokenizer, sos and eos token, better to print out then trunctate
        logits = logits.view(-1, logits.size(-1))  ## format as [bs, num_cls]
        # print("logits.shape: ", logits.shape)


        # Prepare the target tokens (shift answer tokens by one position)
        target_tokens = combine_tokens[:, prompt_len:].squeeze() # Remove batch dimension if necessary
        ### target_tokens = combine_tokens[:, 1:].squeeze() # Remove batch dimension if necessary
        target_tokens = target_tokens.view(-1)   ## format as [bs]
        # print("target_tokens: ", target_tokens.shape, target_tokens)

        # pdb.set_trace()

        # assert False

        # Compute the loss
        loss_fct = nn.CrossEntropyLoss(reduction = "sum")
        loss = loss_fct(logits, target_tokens)

        return loss.item()
    
    return prompt_to_loss
