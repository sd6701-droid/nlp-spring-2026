import numpy as np
import pandas as pd
from tqdm import tqdm
from submission import your_prompt, your_config, your_post_processing, your_pre_processing, your_hf_token, your_netid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import sleep
import ast
import inspect
from sklearn.metrics import mean_absolute_error
from huggingface_hub import login


YOUR_HF_TOKEN = your_hf_token()
login(YOUR_HF_TOKEN)

# --- Globals for the loaded model and device ---
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_id="meta-llama/Llama-2-7b-chat-hf"):
    """Loads a model and tokenizer from Hugging Face if not already loaded."""
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None:
        # Already loaded — just reuse it
        return MODEL, TOKENIZER

    print(f"Loading model: {model_id} onto device: {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    print("Model loaded successfully.")
    return MODEL, TOKENIZER

def contains_addition(node):
    """Recursively check if the given AST node or its children contain an addition operation."""
    if isinstance(node, ast.Add):
        return True
    for child in ast.iter_child_nodes(node):
        if contains_addition(child):
            return True
    return False

def function_uses_addition(func):
    """Check if a function uses arithmetic addition."""
    source = inspect.getsource(func)  # Get the source code of the function
    tree = ast.parse(source)
    return contains_addition(tree)

def get_addition_pairs(lower_bound, upper_bound, rng):
    int_a = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    int_b = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    return int_a, int_b


def call_model(prompt, student_configs, post_processing_fn):
    """Generates a response using the provided local Hugging Face model and tokenizer."""
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        load_model()  # Load once if not already loaded

    # 1. Tokenize the input prompt
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    # Adapt config keys for Hugging Face's `generate`
    hf_configs = student_configs.copy()
    if 'max_tokens' in hf_configs:
        hf_configs['max_new_tokens'] = hf_configs.pop('max_tokens')
    hf_configs.pop('stop', None)  # Remove 'stop' if exists

    # 2. Generate output tokens
    outputs = MODEL.generate(**inputs, **hf_configs)
    
    # 3. Decode generated tokens
    result_new = TOKENIZER.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 4. Apply post-processing
    final_output = post_processing_fn(result_new)
    
    return final_output


def test_range(added_prompt, prompt_configs, rng, n_sample=30, lower_bound=1, upper_bound=10, fixed_pairs=None, 
               pre_processing=your_pre_processing,post_processing=your_post_processing):
    int_as = []
    int_bs = []
    answers = []
    model_responses = []
    correct = []
    prompts = []
    iterations = fixed_pairs if not (fixed_pairs is None) else []
    for _ in range(n_sample):
        int_a, int_b = get_addition_pairs(lower_bound, upper_bound, rng=rng)
        iterations.append((int_a, int_b))
    for i, v in enumerate(tqdm(iterations)):
        int_a, int_b = v
        fixed_prompt = f'{int_a}+{int_b}'
        fixed_prompt = pre_processing(fixed_prompt)
        print(f'added prompt is {added_prompt}')
        prefix, suffix = added_prompt
        prompt = prefix + fixed_prompt + suffix
        model_response = call_model(prompt, prompt_configs, post_processing)
        answer = int_a + int_b
        int_as.append(int_a)
        int_bs.append(int_b)
        prompts.append(prompt)
        answers.append(answer)
        model_responses.append(model_response)
        correct.append((answer == model_response))
        sleep(1) # pause to not trigger DDoS defense
    df = pd.DataFrame({'int_a': int_as, 'int_b': int_bs, 'prompt': prompts, 'answer': answers, 'response': model_responses, 'correct': correct})
    print(df.to_string())
    mae = mean_absolute_error(df['answer'], df['response'])
    acc = df.correct.sum()/len(df)
    prompt_length = len(prefix) + len(suffix)
    res = acc * (1/prompt_length) * (1-mae/(5*(10**6)))
    return {'res': res, 'acc': acc, 'mae': mae, 'prompt_length': prompt_length}