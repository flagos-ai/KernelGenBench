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

import json
import os, sys
import yaml
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Any, List, Union
from sandbox.utils.accuracy_utils import VerifyResult
from pydantic import BaseModel
import torch

import logging

logger = logging.getLogger(__name__)

from .utils import (
    create_inference_server_from_presets,
    extract_first_code,
    maybe_multithread,
    read_file,
    # set_gpu_arch,
    construct_dataset,
    prompt_generate_custom_triton_from_prompt_template,
    prompt_generate_test_func_from_prompt_template
)

# Import new base class and TritonKernelGenerateArgs
from kernelgenbench.framework.generate_args import (
    BaseGenerateArgs as _NewBaseGenerateArgs,
    TritonKernelGenerateArgs as _NewTritonKernelGenerateArgs,
    InputArg as _NewInputArg,
    OutputArg as _NewOutputArg,
)

# Backward compatibility aliases
BaseGenerateArgs = _NewBaseGenerateArgs
TritonKernelGenerateArgs = _NewTritonKernelGenerateArgs
InputArg = _NewInputArg
OutputArg = _NewOutputArg

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


@dataclass
class GenerationConfig:
    run_name: str
    sample_id: int = 0
    num_samples: int = 1
    test_type: str = "accuracy" # accuracy, performance, both
    name_list: Optional[Any] = None
    num_workers: int = 1
    api_query_interval: float = 0.0
    # Default values required for dataclass initialization - override via CLI args
    server_type: str = "openai"
    model_name: str = "gpt-4o"
    max_tokens: int = 16384
    temperature: float = 0.0
    run_dir: str = os.path.join(REPO_TOP_DIR, "runs")
    save_log: bool = True
    verbose: bool = False
    store_type: str = "local"
    log_prompt: bool = False
    backend: str = "triton"
    greedy_sample: bool = False
    base_url: str = "http://localhost:8000/v1"
    strict_check: bool = False
    seed: int = 42
    use_ai_advice: bool = False

    def set_greedy(self):
        self.greedy_sample = True

@dataclass
class WorkArgs:
    problem_id: int  # logically indexed
    sample_id: int
    op_name: str | None = None
    op_obj: Callable | None = None

class ConstructDataArgs(BaseModel):
    kernel_bench_code: str
    problem_name: str

    @property
    def op_name(self):
        return self.problem_name

# TorchKernelGenerateArgs, TestFuncGenerateArgs, BenchmarkFuncGenerateArgs are kept here
# because they have not yet been migrated to the framework module
class TorchKernelGenerateArgs(BaseGenerateArgs):
    torch_kernel_name: str
    func_desc: str
    input_args: List[InputArg] | None = None
    output_args: List[OutputArg] | None = None
    func_type: Optional[str] = None  # "unary", "binary", "reduction", "other"

    @property
    def op_name(self):
        return self.torch_kernel_name

# class TestFuncGenerateArgs(BaseGenerateArgs):
#     test_func_name: str
#     torch_kernel_name: str
#     triton_kernel_name: str
#     torch_kernel_code: str
#     triton_kernel_code: str | None = None

#     @property
#     def op_name(self):
#         return self.test_func_name
class TestFuncGenerateArgs(BaseGenerateArgs):
    test_func_name: str
    kernel_name: str
    operators: Dict[str, str]
    ops_namespace: str | None = None

    @property
    def op_name(self):
        return self.test_func_name

class BenchmarkFuncGenerateArgs(BaseGenerateArgs):
    test_perf_func_name: str
    kernel_name: str
    test_func_code: str
    operators: Dict[str, str]
    ops_namespace: str | None = None

    @property
    def op_name(self):
        return self.test_perf_func_name

def generate_sample_single(
    work: Union[WorkArgs, TestFuncGenerateArgs, TritonKernelGenerateArgs, TorchKernelGenerateArgs, BenchmarkFuncGenerateArgs],
    config: GenerationConfig,
    # dataset,
    inference_server: Callable,
    run_dir: str,
    return_type: str = "return", # save, return, both
    check_result: VerifyResult | None = None,
    prompt_fn: Callable | None = None,
) -> bool:
    if prompt_fn is None:
        prompt_fn = prompt_generate_custom_triton_from_prompt_template
    custom_cuda_prompt = prompt_fn(
        work,
    )
    if not custom_cuda_prompt:
        logger.warning(f"Prompt generation failed for {work.op_name}, skipping...")
        return False, work, ""
    if config.log_prompt:
        prompt_path = os.path.join(
            run_dir,
            f"prompt_{config.sample_id}",
            f"problem_{work.op_name}_sample_{config.sample_id}_prompt.txt",
        )
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    # extracted_custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # if extracted_custom_cuda is not None:
    #     custom_cuda = extracted_custom_cuda
    # else:
    #     logger.warning(f"Code extraction failed for {work.op_name}, using raw output.")
    # # check LLM is able to generate custom CUDA code
    # assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        logger.info(
            f"Generated sample {work.sample_id} for problem {work.problem_id}: {work.op_name}"
        )

    # Store to local file
    if return_type in ["save", "both"]:
        kernel_path = os.path.join(
            run_dir,
            f"code_{work.sample_id}", 
            f"{work.op_name}.py",
        )
        os.makedirs(os.path.dirname(kernel_path), exist_ok=True)
        with open(kernel_path, "w") as f:
            f.write(custom_cuda)
    if return_type in ["both", "return"]:
        return True, work, custom_cuda
    else:
        return True


def generate_sample_launcher(
    work: Union[WorkArgs, TestFuncGenerateArgs, TritonKernelGenerateArgs, TorchKernelGenerateArgs],
    config: GenerationConfig,
    inference_server: Callable,
    run_dir: str,
    check_result: VerifyResult | None = None,
    **kwargs
):
    try:
        return generate_sample_single(work, config, inference_server, run_dir, check_result=check_result, **kwargs)
    except Exception as e:
        logger.error(f"Error generating sample {work.op_name} {config.sample_id}: {e}")
        return False, work, ""


def check_kernel_exists(
    run_dir: str, level: int, problem_id: int, sample_id: int
) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


def sample(config: GenerationConfig, check_result: List[VerifyResult] | None = None):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    logger.info(f"Starting Batch Generation with config: {config}")

    curr_level_dataset = construct_dataset()
    
    if config.name_list is not None:
        curr_level_dataset = {
            key: curr_level_dataset.get(key, None) for key in config.name_list
        }

    # set up run directory
    run_dir = os.path.join(config.run_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "code"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "prompt"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "log"), exist_ok=True)

    with open(os.path.join(run_dir, "generation_config.yaml"), "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

    assert (
        config.store_type == "local"
    ), "supporting local file-system based storage for now"  # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    for problem_id, (op_name, op_obj) in enumerate(curr_level_dataset.items()):  # end index is inclusive
        problems_to_run += [
            WorkArgs(
                problem_id=problem_id + 1, 
                sample_id=config.sample_id + sample_idx, 
                op_name=op_name, 
                op_obj=op_obj
            ) for sample_idx in range(config.num_samples)
        ]
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        base_url=config.base_url,
    )

    # Launch workers
    generation_results = maybe_multithread(
        func=generate_sample_launcher,
        instances={"work": problems_to_run, "check_result": check_result} if check_result is not None else problems_to_run, 
        num_workers=config.num_workers,
        time_interval=config.api_query_interval,
        # extra args
        config=config,
        # dataset=curr_level_dataset,
        inference_server=inference_server,
        run_dir=run_dir,
    )

    num_generated_samples = len(generation_results)
    total_problems = len(problems_to_run)
    num_failed_problems = total_problems - num_generated_samples
    logger.info(
        f"Generated {num_generated_samples} samples for total {total_problems} problems, Please retry for the {num_failed_problems} failed problems."
    )
    return generation_results
