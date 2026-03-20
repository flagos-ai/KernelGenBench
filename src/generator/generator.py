from .sampler.utils import (
    create_inference_server_from_presets, 
    maybe_multithread, 
)
from .sampler.generate_samples import (
    generate_sample_launcher,
)
from .sampler import GenerationConfig
import os
from typing import Callable

from rich.console import Console
console = Console()

# a decoretor to print the prompt
def print_prompt(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        prompt = func(*args, **kwargs)
        console.rule("[bold blue]Generated Prompt")
        console.print(prompt)
        console.rule("[bold blue]End of Prompt")
        return prompt
    return wrapper

class BaseGenerator:
    def __init__(self, generation_config: GenerationConfig):
        self.generation_config = generation_config
        self.inference_server = create_inference_server_from_presets(
            server_type=generation_config.server_type,
            model_name=generation_config.model_name,
            temperature=generation_config.temperature,
            max_tokens=generation_config.max_tokens,
            verbose=generation_config.verbose,
            base_url=generation_config.base_url,
        )
        self.from_mcp = False

    def __call__(self, kwargs, on_result=None) -> list:
        if not isinstance(kwargs, list):
            kwargs = [kwargs]
        kwargs = [self._init_data(kwarg) for kwarg in kwargs]
        kwargs = {
            "work": kwargs,
            "prompt_fn": [self.generate_prompt for _ in range(len(kwargs))],
        }
        # kwargs.update({"prompt_fn": self.generate_prompt})
        generation_results = maybe_multithread(
            generate_sample_launcher,
            instances=kwargs,
            num_workers=self.generation_config.num_workers,
            time_interval=self.generation_config.api_query_interval,
            on_result=on_result,
            config=self.generation_config,
            inference_server=self.inference_server,
            run_dir=os.path.join(self.generation_config.run_dir, self.generation_config.run_name)
        )
        generation_results = self.post_process(generation_results)
        return generation_results


    def generate_prompt(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _init_data(self, kwargs):
        # raise NotImplementedError("Subclasses should implement this method.")
        return kwargs
    
    def post_process(self, results: list) -> list:
        codes = [r[-1] for r in results]
        return codes