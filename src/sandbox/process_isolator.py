"""
Layer 5: Process Isolation Execution.
From: triton_competition_anti_cheat_guide.md - Section 7
"""
import multiprocessing as mp
import os
import time
from typing import Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class TestConfig:
    """测试配置"""
    seed: int
    shape: tuple
    dtype: str
    noise_config: dict
    layout_config: dict


def isolated_test_worker(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    隔离进程工作函数

    每次调用都是全新进程，清理所有全局状态
    """
    import torch
    import random

    # ===== 第一步：文件系统隔离 =====
    from sandbox.cache_isolator import CacheIsolator
    cache_isolator = CacheIsolator()
    cache_isolator.isolate()

    # ===== 第二步：环境变量设置 =====
    os.environ['TRITON_DISABLE_AUTOTUNE'] = '1'
    os.environ['TORCHINDUCTOR_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '1'

    # ===== 第三步：启用Import Hook沙箱 =====
    from sandbox.import_hook import RuntimeSandbox
    sandbox = RuntimeSandbox()
    sandbox.enable()

    try:
        # ===== 第四步：CUDA层保护 =====
        from sandbox.cuda_protector import CUDALayerProtector
        cuda_protector = CUDALayerProtector()
        cuda_protector.setup()

        # ===== 第五步：随机种子设置 =====
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])

        # ===== 第六步：准备输入 =====
        M, N, K = config['shape']

        dtype = getattr(torch, config['dtype'])
        runs = config.get('runs', 4)
        per_iteration_seeds = config.get('per_iteration_seeds',
                                         [config['seed'] + i for i in range(runs)])

        from sandbox.shape_generator import TensorLayoutRandomizer
        layout_randomizer = TensorLayoutRandomizer()

        # ===== 第七步：加载并执行算子 =====
        operator = config['load_operator']()

        # Warmup with dedicated seed
        warmup_seed = config.get('warmup_seed', config['seed'] - 1)
        torch.manual_seed(warmup_seed)
        random.seed(warmup_seed)
        w_input = torch.randn((M, K), dtype=dtype, device='cuda').clone().contiguous()
        w_weight = torch.randn((K, N), dtype=dtype, device='cuda').clone().contiguous()
        for _ in range(config.get('warmup', 1)):
            _ = operator(w_input.clone(), w_weight.clone())
        torch.cuda.synchronize()

        # 正式计时：每轮不同 seed + clone 杀死缓存
        times = []
        for iter_seed in per_iteration_seeds:
            torch.manual_seed(iter_seed)
            random.seed(iter_seed)

            input_tensor = torch.randn((M, K), dtype=dtype, device='cuda').clone().contiguous()
            weight = torch.randn((K, N), dtype=dtype, device='cuda').clone().contiguous()

            if config.get('randomize_layout', True):
                weight = layout_randomizer.randomize_layout(weight)

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = operator(input_tensor.clone(), weight.clone())
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
            del input_tensor, weight

        return {
            'times': times,
            'shape': (M, N, K),
            'seed': config['seed'],
            'status': 'success'
        }

    except Exception as e:
        return {
            'error': str(e),
            'status': 'failed'
        }

    finally:
        # 清理
        sandbox.disable()
        cache_isolator.cleanup()


class ProcessIsolatedEvaluator:
    """进程隔离评测器"""

    def __init__(self, num_workers: int = 1):
        self.ctx = mp.get_context('spawn')

    def evaluate_single(self,
                       operator_loader: Callable,
                       test_config: TestConfig) -> Dict[str, Any]:
        """单次评测"""
        config = {
            'seed': test_config.seed,
            'shape': test_config.shape,
            'dtype': test_config.dtype,
            'noise_config': test_config.noise_config,
            'layout_config': test_config.layout_config,
            'load_operator': operator_loader,
            'warmup': 1,
            'runs': 4,
            'randomize_layout': True,
        }

        # 序列化config（ multiprocessing需要）
        # 注意：operator_loader需要能被pickle

        with self.ctx.Pool(1) as pool:
            result = pool.apply(isolated_test_worker, (config,))

        return result

    def evaluate_batch(self,
                      operator_loader: Callable,
                      test_configs: list) -> list:
        """批量评测（每组独立进程）"""
        results = []
        for config in test_configs:
            result = self.evaluate_single(operator_loader, config)
            results.append(result)
        return result
