# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import time
import concurrent
import logging

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

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

PROBLEM_FIX_STATEMENT = """
You are a Triton kernel expert. I have a Triton code snippet that needs fixing. The purpose of this code is to implement a kernel in Triton that has the same functionality as the original Torch API. I will provide you with the faulty code along with the corresponding traceback.
"""

PROBLEM_FIX_INSTRUCTION = """
Fix the Triton code snippet. The code snippet should implement the same functionality as the original Torch API. The code snippet should be wrapped in a single code block using triple backticks. Do not include any testing code, comments, explanations, or any other text. Output only valid, executable code.
"""

########################################################
# Inference Helpers
########################################################

def query_server(
    prompt: str | list[dict],
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 128,
    num_completions: int = 1,
    server_type: str = "anthropic",
    model_name: str = "default",
    is_reasoning_model: bool = False,
    budget_tokens: int = 0,
    reasoning_effort: str = None,
    base_url: str = None,
    **kwargs,
):
    if server_type not in {"anthropic", "openai"}:
        raise ValueError(
            f"Unsupported API format: {server_type!r}. "
            "Use 'openai' for OpenAI-compatible APIs or 'anthropic' for "
            "Anthropic-compatible APIs."
        )

    match server_type:
        case "anthropic":
            import anthropic as _anthropic
            client_args = {}
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
            if api_key:
                client_args["api_key"] = api_key
            if resolved_base_url:
                client_args["base_url"] = resolved_base_url
            client = _anthropic.Anthropic(**client_args)
            model = model_name
        case "openai":
            client_args = {}
            api_key = os.environ.get("OPENAI_API_KEY")
            resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if api_key:
                client_args["api_key"] = api_key
            if resolved_base_url:
                client_args["base_url"] = resolved_base_url
            client = OpenAI(**client_args)
            model = model_name

    if server_type == "anthropic":
        assert type(prompt) == str
        if is_reasoning_model:
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        outputs = [choice.text for choice in response.content if not hasattr(choice, 'thinking') or not choice.thinking]
    else:
        messages = prompt if isinstance(prompt, list) else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if is_reasoning_model:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = [choice.message.content for choice in response.choices]

    return outputs[0] if len(outputs) == 1 else outputs


SERVER_PRESETS = {
    "anthropic": {
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "temperature": 0.0,
        "max_tokens": 4096,
    },
}


def create_inference_server_from_presets(server_type: str = None,
                                         greedy_sample: bool = False,
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs) -> callable:
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS.get(server_type, {}).copy()
        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            logger.info(f"Querying server {server_type} with args: {server_args}")
        if time_generation:
            start = time.time()
            response = query_server(prompt, server_type=server_type, **server_args)
            logger.info(f"[Timing] Inference took {time.time() - start:.2f} seconds")
            return response
        return query_server(prompt, server_type=server_type, **server_args)
    return _query_llm


########################################################
# Dataset
########################################################

from kernelgenbench.dataset import get_kernelgenbench_operators
from sandbox.utils.accuracy_utils import VerifyResult
import importlib


def construct_dataset():
    return get_kernelgenbench_operators()


########################################################
# File utils
########################################################


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        logger.info(f"File {file_path} does not exist")
        return ""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def get_torch_op_docstring(torch_op: str) -> str:
    try:
        parts = torch_op.split(".")
        module = importlib.import_module(parts[0])
        obj = module
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj.__doc__ or ""
    except (ImportError, AttributeError) as e:
        return f"Cannot find docstring for '{torch_op}': {e}"


FEW_SHOTS_OP = {
    "torch.nn.functional.scaled_dot_product_attention": "",
}


def prompt_generate_custom_triton_from_prompt_template(torch_op: str, check_result: VerifyResult = None, pointwise: bool = False) -> str:
    if pointwise:
        raise NotImplementedError("Pointwise Triton generation not implemented yet")
    if check_result is not None and check_result.success == False:
        prompt = PROBLEM_FIX_STATEMENT
        prompt += "The original torch operator is:\n```python\n" + torch_op + "\n```\n"
        prompt += "The original triton kernel is:\n```python\n" + check_result.triton_code + "\n```\n"
        prompt += "The original torch operator's input is:\n```python\n" + str(check_result.params) + "\n```\n"
        prompt += "The original torch operator's traceback is:\n```python\n" + check_result.traceback + "\n```\n"
        prompt += PROBLEM_FIX_INSTRUCTION
    else:
        prompt = PROBLEM_STATEMENT
        prompt += PROBLEM_INSTRUCTION
        prompt += f"You are given the torch operator \"{torch_op}\". Please write a custom Triton kernel for this operator that fully replicates the behavior of the corresponding PyTorch operator, including support for the same input shapes, data types, broadcasting semantics, and numerical precision, so that it can be used as a drop-in replacement in PyTorch workflows.\n"
        try:
            docstring = get_torch_op_docstring(torch_op)
            if docstring:
                prompt += f"The torch api's signature is:\n```python\n{docstring}\n```\n"
        except Exception:
            pass
    return prompt


def prompt_generate_test_func_from_prompt_template(torch_op: str) -> str:
    return (
        "You are a testing expert. You are given a PyTorch operator and a custom Triton kernel that implements "
        "the same functionality as the PyTorch operator. Your task is to write a comprehensive test function in Python "
        "that verifies the correctness of the Triton kernel by comparing its output against the PyTorch operator's output "
        "across a variety of input scenarios.\n"
    )


########################################################
# Code extraction
########################################################

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    if output_string is None:
        return None
    trimmed = output_string.strip()
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()
        return code
    return None


########################################################
# Parallel execution
########################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, **shared_kwargs):
    output_data = []
    if isinstance(instances, dict):
        instances = [dict(zip(instances.keys(), values)) for values in zip(*instances.values())]
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for instance in instances:
                    if isinstance(instance, dict):
                        futures.append(executor.submit(func, **instance, **shared_kwargs))
                    else:
                        futures.append(executor.submit(func, instance, **shared_kwargs))
                    time.sleep(time_interval)
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        logger.error("Got an error!", e)
    else:
        for instance in tqdm(instances):
            output = func(**instance, **shared_kwargs) if isinstance(instance, dict) else func(instance, **shared_kwargs)
            if output is not None:
                output_data.append(output)
    return output_data
