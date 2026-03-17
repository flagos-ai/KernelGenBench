########################
# Utils Functions
########################

import multiprocessing
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import importlib
import os
import json
from tqdm import tqdm

# API clients
# from together import Together
from openai import OpenAI
# import google.generativeai as genai
# import anthropic

# from datasets import load_dataset
import numpy as np
import torch
from contextlib import contextmanager
from collections import defaultdict
import time
import shutil
import concurrent
from functools import cache
# from transformers import AutoTokenizer
import hashlib
# from bench import PYTORCH_OPERATORS
from flagbench.dataset.kernel_list import PYTORCH_OPERATORS
from sandbox.utils.accuracy_utils import VerifyResult

from concurrent.futures import ProcessPoolExecutor, as_completed

import logging

logger = logging.getLogger(__name__)

# Define API key access
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
PANDA_API_KEY = os.environ.get("PANDA_API_KEY")
KSYUN_API_KEY = os.environ.get("KSYUN_API_KEY")

############################################
# Triton Prompt
############################################

PROBLEM_STATEMENT = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. \n
You have complete freedom to choose the set of operators you want to replace. \
You may make the decision to replace some operators with custom Triton kernels \
and leave others unchanged. You may replace multiple operators with custom \
implementations, consider operator fusion opportunities (combining multiple \
operators into a single kernel, for example, combining matmul+relu), or algorithmic \
changes (such as online softmax). You are only limited by your imagination.\n
"""

# PROBLEM_INSTRUCTION = """
# Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
# """
PROBLEM_INSTRUCTION = """
Rewrite the following PyTorch operator using custom Triton kernels. Replace the \
operator logic entirely with Triton code. Wrap your output in a single code block \
using triple backticks. Do not include any testing code, comments, explanations, \
or any other text. Output only valid, executable code. You can customize the name \
of your kernel function, but the final function name must be consistent with the \
corresponding PyTorch function. Do not implement any registration functionality, only implement the core functionality.

IMPORTANT REQUIREMENTS:
1. For pointwise operators, you MUST handle broadcasting correctly. Ensure your kernel \
supports inputs with different shapes that can be broadcast together according to PyTorch's \
broadcasting semantics.
2. You MUST handle non-contiguous tensors correctly. Do not assume input tensors are \
contiguous in memory. Use proper stride calculations to access elements correctly for \
tensors with arbitrary memory layouts.
"""


FEW_SHOTS_OP = {
    "torch.nn.functional.scaled_dot_product_attention": "/share/project/tj/workspace/attention.py",
}


PROBLEM_FIX_STATEMENT = """
You are a Triton kernel expert. I have a Triton code snippet that needs fixing. The purpose of this code is to implement a kernel in Triton that has the same functionality as the original Torch API. I will provide you with the faulty code along with the corresponding traceback. 
"""

PROBLEM_FIX_INSTRUCTION = """
Fix the Triton code snippet. The code snippet should implement the same functionality as the original Torch API. The code snippet should be wrapped in a single code block using triple backticks. Do not include any testing code, comments, explanations, or any other text. Output only valid, executable code.
"""

########################################################
# Inference Helpers
########################################################

# @cache
# def load_deepseek_tokenizer():
#     # TODO: Should we update this for new deepseek? Same tokenizer?
#     # return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Instruct-0724")
#     return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000


# def is_safe_to_send_to_deepseek(prompt):
#     tokenizer = load_deepseek_tokenizer()
#     # print(f"Prompt: {len(prompt)}")
#     # print(f"Prompt length: {len(tokenizer(prompt, verbose=False)['input_ids'])}")
    
#     if type(prompt) == str:
#         return (
#             len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
#         )
#     else:
#         return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK

def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

def query_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_port: int = 30000,  # only for local server hosted on SGLang
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type

    # for reasoning models
    is_reasoning_model: bool = False, # indiactor of using reasoning models
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = None, # only for o1 and o3 / more reasoning models in the future
    base_url: str = None, # only for vllm local server
):
    """
    Query various sort of LLM inference API providers
    Supports:
    - OpenAI
    - Deepseek
    - Together
    - Sambanova
    - Anthropic
    - Gemini / Google AI Studio
    - Fireworks (OpenAI compatbility)
    - SGLang (Local Server)
    """
    # Select model and client based on arguments
    match server_type:
        # case "sglang":
        #     url = f"http://{server_address}:{server_port}"
        #     client = OpenAI(
        #         api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
        #     )
        #     model = "default"
        # case "deepseek":
        #     client = OpenAI(
        #         api_key=DEEPSEEK_KEY,
        #         base_url="https://api.deepseek.com",
        #         timeout=10000000,
        #         max_retries=3,
        #     )
        #     model = model_name
        #     assert model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"], "Only support deepseek-chat or deepseek-coder for now"
        #     if not is_safe_to_send_to_deepseek(prompt):
        #         raise RuntimeError("Prompt is too long for DeepSeek")
        # case "fireworks":
        #     client = OpenAI(
        #         api_key=FIREWORKS_API_KEY,
        #         base_url="https://api.fireworks.ai/inference/v1",
        #         timeout=10000000,
        #         max_retries=3,
        #     )
        #     model = model_name

        # case "anthropic":
        #     client = anthropic.Anthropic(
        #         api_key=ANTHROPIC_KEY,
        #     )
        #     model = model_name
        # case "google":
        #     genai.configure(api_key=GEMINI_KEY)
        #     model = model_name
        # case "together":
        #     client = Together(api_key=TOGETHER_KEY)
        #     model = model_name
        # case "sambanova":
        #     client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
        #     model = model_name
            
        # case "openai":
        #     client = OpenAI(api_key=OPENAI_KEY)
        #     model = model_name

        # case "qwen":
        #     client = OpenAI(
        #         api_key=DASHSCOPE_API_KEY, 
        #         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        #     )
        #     model = model_name
        
        # case "vllm":
        #     client = OpenAI(
        #         api_key="EMPTY", 
        #         base_url=base_url if base_url else "http://localhost:8000/v1"
        #     )
        #     model = model_name

        case "panda":
            client = OpenAI(
                api_key=PANDA_API_KEY,
                base_url="https://api.pandalla.ai/v1",
                timeout=10000000,
                max_retries=10,
            )
            model = model_name

        case "ksyun":
            client = OpenAI(
                api_key=KSYUN_API_KEY,
                base_url="https://kspmas.ksyun.com/v1",
                timeout=10000000,
                max_retries=10,
            )
            # Model name mapping for ksyun
            ksyun_model_mapping = {
                "gpt-5": "mog-1",
                "gpt-5.2": "mog-2",
                "gemini-3-pro-preview": "mgg-2",
                "gemini-3-flash": "mgg-7",
            }
            model = ksyun_model_mapping.get(model_name, model_name)

        case _:
            raise NotImplementedError

    if server_type != "google":
        assert client is not None, "Client is not set, cannot proceed to generations"
    else:
        logger.info(
            f"Querying {server_type} {model} with temp {temperature} max tokens {max_tokens}"
        )
    # Logic to query the LLM
    if server_type == "qwen":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            # extra_body={"enable_thinking": False}, 
            stream=False, 
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "vllm":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "panda":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "ksyun":
        ksyun_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
        )
        # Claude models don't support both temperature and top_p
        _no_top_p_models = ("mco", "ep-20260313004351-feb36", "ep-20260317033721-n8ndf")
        if not model.startswith(_no_top_p_models):
            ksyun_kwargs["top_p"] = top_p
        response = client.chat.completions.create(**ksyun_kwargs)
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "anthropic":
        assert type(prompt) == str

        if is_reasoning_model:
            # Use beta endpoint with thinking enabled for reasoning models
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                # Claude thinking requires budget_tokens for thinking (reasoning)
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            # Use standard endpoint for normal models
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        outputs = [choice.text for choice in response.content if not hasattr(choice, 'thinking') or not choice.thinking]

    elif server_type == "google":
        # assert model_name == "gemini-1.5-flash-002", "Only test this for now"

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        response = model.generate_content(prompt)

        return response.text

    elif server_type == "deepseek":
        
        if model in ["deepseek-chat", "deepseek-coder"]:
            # regular deepseek model 
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        else: # deepseek reasoner
            # assert is_reasoning_model, "Only support deepseek-reasoner for now"
            assert model == "deepseek-reasoner", "Only support deepseek-reasoner for now"
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                n=num_completions,
                max_tokens=max_tokens,
                # do not use temperature or top_p
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "openai":
        if is_reasoning_model:
            assert "o1" in model or "o3" in model, "Only support o1 and o3 for now"
            logger.info(f"Using OpenAI reasoning model: {model} with reasoning effort {reasoning_effort}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort=reasoning_effort,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "together":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
            top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "fireworks":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # top_p=top_p,
            # top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "sambanova":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    # for all other kinds of servers, use standard API
    else:
        if type(prompt) == str:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]
        else:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.message.content for choice in response.choices]

    # output processing
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek",
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini-1.5-flash-002",
        "temperature": 0.7, # need to experiment with temperature
        "max_tokens": 8192,
    },
    "together": { # mostly for Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is for running locally, mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 10210,
        "server_address": "matx2.stanford.edu",
        "max_tokens": 8192,
    },
    "anthropic": {  # for Claude 3.5 Sonnet
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "sambanova": {
        "model_name": "Meta-Llama-3.1-405B-Instruct",
        "temperature": 0.1,
        "max_tokens": 8192,
    },
    "qwen": {
        "temperature": 0.2, 
        "model_name": "qwen3-coder-plus",
        "max_tokens": 4096
    }, 
    "vllm": {
        "model_name": "Qwen3-Coder-30B-A3B-Instruct/",
        "temperature": 0.0,
        "max_tokens": 32768,
    }, 
    "panda": {
        "model_name": "claude-opus-4-1-20250805",
        "temperature": 0.0,
        "max_tokens": 32768,
    },
    "ksyun": {
        "model_name": "glm-4.7",
        "temperature": 0.0,
        "max_tokens": 32768,
    }
}


def create_inference_server_from_presets(server_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            logger.info(f"Querying server {server_type} with args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            logger.info(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, server_type=server_type, **server_args
            )
    
    return _query_llm

"""
Model output processing
#  TODO: add unit tests
"""



def print_messages(messages):
    for message in messages:
        logger.info(message["role"])
        logger.info(message["content"])
        logger.info("-" * 50)
        logger.info("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    if output_string is None:
        return None

    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""

################################################################################
# Scale up experiments in parallel
################################################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, **shared_kwargs):
    """
    Multithreaded execution of func, with optional time interval between queries
    Ideal for querying LLM APIs, does not provide process isolation
    """
    output_data = []
    if isinstance(instances, dict):
        instances = [dict(zip(instances.keys(), values)) for values in zip(*instances.values())]
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

                # Submit tasks one at a time with delay between them
                futures = []
                for instance in instances:
                    if isinstance(instance, dict):
                        futures.append(
                            executor.submit(
                                func,
                                **instance,
                                **shared_kwargs
                            )
                        )
                    else:
                        futures.append(
                            executor.submit(
                                func,
                                instance,
                                **shared_kwargs
                            )
                        )
                    time.sleep(time_interval)  # sleep between submitting each task



                # Wait for each future to complete
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        logger.error("Got an error!", e)
                        continue
    else:
        for instance in tqdm(instances):
            output = func(**instance, **shared_kwargs) if isinstance(instance, dict) else func(instance, **shared_kwargs)
            if output is not None: output_data.append(output)

    return output_data


def maybe_multiprocess_cuda(
    func, instances, num_workers, *shared_args, **shared_kwargs
):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(func, instance, *shared_args, **shared_kwargs): None
                for instance in instances
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None:
                        output_data.append(result)
                except Exception as e:
                    logger.error("Got an error!", e)
                    continue
    return output_data


def get_src_from_torch_op(torch_op: str) -> str:
    """
    Get the source code of a PyTorch operator by its name.
    This is useful for generating custom Triton kernels from PyTorch operators.
    """
    try:
        # Get the source code of the PyTorch operator
        src = torch_op.__code__.co_code
        return src
    except Exception as e:
        # print(f"Error getting source code for {torch_op}: {e}")
        logger.warning(f"Error getting source code for {torch_op}: {e}")
        return ""


def read_file(file_path: str) -> str:
    """
    Read the content of a file and return it as a string.
    If the file does not exist or cannot be read, return an empty string.
    """
    if not os.path.exists(file_path):
        logger.info(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def prompt_generate_custom_triton_from_prompt_template(torch_op: str, check_result: VerifyResult = None, pointwise: bool = False) -> str:
    #TODO: get few shot source code from torch
    # few_shot_src = []
    # if FEW_SHOTS_OP:
    #     for op in FEW_SHOTS_OP:
    #         few_shot_src.append(get_src_from_torch_op(op))
    if pointwise:
        raise NotImplementedError("Pointwise Triton generation not implemented yet")
    if check_result is not None and check_result.success == False:
        prompt = PROBLEM_FIX_STATEMENT
        prompt += "The original torch operator is:\n```python\n"
        prompt += torch_op
        prompt += "\n```\n"
        prompt += "The original triton kernel is:\n```python\n"
        prompt += check_result.triton_code
        prompt += "\n```\n"
        prompt += "The original torch operator's input is:\n```python\n"
        prompt += str(check_result.params)
        prompt += "\n```\n"
        prompt += "The original torch operator's traceback is:\n```python\n"
        prompt += check_result.traceback
        prompt += "\n```\n"
        prompt += PROBLEM_FIX_INSTRUCTION
    else:
        # Generate the prompt for Triton kernel generation
        prompt = PROBLEM_STATEMENT
        if FEW_SHOTS_OP:
            prompt += "Here's some examples to show you how to write custom Triton kernels:\n"
            for op in FEW_SHOTS_OP:
                prompt += f"- {op.__str__}\n"
                src = get_src_from_torch_op(op)
                if src:
                    prompt += f"here is the torch source code for {op}:\n```python\n{src}\n```\n"
                triton_src = read_file(FEW_SHOTS_OP[op])
                prompt += "the example triton kernel code is:\n```python\n" + triton_src + "\n```\n"
        prompt += PROBLEM_INSTRUCTION
        prompt += f"You are given the torch operator \"{torch_op}\". Please write a custom Triton kernel for this operator that fully replicates the behavior of the corresponding PyTorch operator, including support for the same input shapes, data types, broadcasting semantics, and numerical precision, so that it can be used as a drop-in replacement in PyTorch workflows.\n"
        try:
            docstring = get_torch_op_docstring(torch_op)
            if docstring:
                prompt += f"The torch api's signature is:\n```python\n{docstring}\n```\n"
        except:
            pass
        
    return prompt

def prompt_generate_test_func_from_prompt_template(torch_op: str, ) -> str:
    #TODO
    prompt = "You are a testing expert. You are given a PyTorch operator and a custom Triton kernel that implements \
    the same functionality as the PyTorch operator. Your task is to write a comprehensive test function in Python \
    that verifies the correctness of the Triton kernel by comparing its output against the PyTorch operator's output \
    across a variety of input scenarios.\n"


def get_torch_op_docstring(torch_op: str) -> str:
    """
    Get the docstring of a PyTorch operator by its full path string.
    Supports module-level functions, classes, and class instance methods.
    """
    # if torch_op == "torch.nn.functional.avg_pool2d":
    #     return torch.nn.functional.avg_pool2d.__doc__ + "\nAnd \'torch.nn.AvgPool2d\''s docstring is:\n" + torch.nn.AvgPool2d.__doc__
    try:
        parts = torch_op.split(".")
        # 动态导入顶层模块
        module = importlib.import_module(parts[0])
        obj = module
        # 依次获取每一层属性
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj.__doc__ or ""
    except (ImportError, AttributeError) as e:
        return f"Cannot find docstring for '{torch_op}': {e}"

def construct_dataset():
    return PYTORCH_OPERATORS