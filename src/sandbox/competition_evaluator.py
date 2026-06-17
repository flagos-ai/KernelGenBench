"""
Layer 9: Complete Competition Evaluation System (S-Level).
From: triton_competition_anti_cheat_guide.md - Section 9
"""
import os
import sys
import time
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
import json


# ============================================================
# 配置
# ============================================================

@dataclass
class CompetitionConfig:
    """竞赛配置"""
    num_tests: int = 10
    warmup_runs: int = 1
    timing_runs: int = 4
    retest_ratio: float = 0.2  # 20%用例需要复测
    cv_threshold: float = 0.15
    iqr_threshold: float = 0.3
    allowed_dtypes: List[str] = None
    shape_categories: List[str] = None

    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = ['float32', 'float16']
        if self.shape_categories is None:
            self.shape_categories = ['small', 'medium', 'large']


# ============================================================
# 测试用例生成
# ============================================================

@dataclass
class TestCase:
    """测试用例"""
    id: int
    seed: int
    shape: tuple
    dtype: str
    layout_type: str
    contiguity: str


class TestCaseGenerator:
    """测试用例生成器"""

    def __init__(self, config: CompetitionConfig):
        self.config = config
        from sandbox.shape_generator import BucketedShapeGenerator, TensorLayoutRandomizer
        self.shape_gen = BucketedShapeGenerator()
        self.layout_gen = TensorLayoutRandomizer()

    def generate_batch(self, num_cases: int) -> List[TestCase]:
        """生成一批测试用例"""
        cases = []
        for i in range(num_cases):
            case = self._generate_single(i)
            cases.append(case)
        return cases

    def _generate_single(self, case_id: int) -> TestCase:
        """生成单个测试用例"""
        # 随机种子
        seed = random.randint(0, 2**31 - 1)

        # 分桶随机shape
        size_category = random.choice(self.config.shape_categories)
        shape = self.shape_gen.generate_gemm_shape(
            size_category=size_category,
            noise_enabled=True
        )

        # 随机dtype
        dtype = random.choice(self.config.allowed_dtypes)

        # 随机layout
        layout_type = random.choice(['contiguous', 'strided', 'transposed'])
        contiguity = random.choice(['contiguous', 'non-contiguous'])

        return TestCase(
            id=case_id,
            seed=seed,
            shape=shape,
            dtype=dtype,
            layout_type=layout_type,
            contiguity=contiguity
        )


# ============================================================
# 评测执行
# ============================================================

def isolated_worker_main(config_dict: dict) -> dict:
    """
    隔离进程主函数

    所有防护措施在此函数内一次性应用
    """
    import torch
    import random
    import numpy as np

    # ===== 第一层：文件系统隔离 =====
    from sandbox.cache_isolator import CacheIsolator
    cache_isolator = CacheIsolator()
    isolated_home = cache_isolator.isolate()

    # ===== 第二层：环境变量 =====
    os.environ['HOME'] = isolated_home
    os.environ['TRITON_DISABLE_AUTOTUNE'] = '1'
    os.environ['TORCHINDUCTOR_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['XDG_CACHE_HOME'] = f'{isolated_home}/.cache'

    # ===== 第三层：Import Hook沙箱 =====
    from sandbox.import_hook import RuntimeSandbox
    sandbox = RuntimeSandbox()
    sandbox.enable()

    # ===== 第四层：CUDA保护 =====
    from sandbox.cuda_protector import CUDALayerProtector
    cuda_protector = CUDALayerProtector()
    cuda_protector.setup()

    try:
        M, N, K = config_dict['shape']
        dtype = getattr(torch, config_dict['dtype'])
        per_iteration_seeds = config_dict.get('per_iteration_seeds',
                                              [config_dict['seed'] + i for i in range(config_dict['timing_runs'])])

        # 加载算子
        operator_path = config_dict['operator_path']
        spec = __import__('importlib.util').util.spec_from_file_location(
            "operator", operator_path
        )
        module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operator = module.forward

        from sandbox.shape_generator import TensorLayoutRandomizer
        layout_randomizer = TensorLayoutRandomizer()

        # Warmup with a dedicated seed (not used in scoring)
        torch.manual_seed(config_dict.get('warmup_seed', config_dict['seed'] - 1))
        random.seed(config_dict.get('warmup_seed', config_dict['seed'] - 1))
        warmup_input = torch.randn((M, K), dtype=dtype, device='cuda').clone().contiguous()
        warmup_weight = torch.randn((K, N), dtype=dtype, device='cuda').clone().contiguous()
        for _ in range(config_dict['warmup_runs']):
            _ = operator(warmup_input.clone(), warmup_weight.clone())
        torch.cuda.synchronize()

        # 正式计时：每次迭代不同 seed，杀死缓存攻击
        times = []
        for i, iter_seed in enumerate(per_iteration_seeds):
            # 每次迭代用不同的 seed → 不同的输入值
            torch.manual_seed(iter_seed)
            random.seed(iter_seed)
            np.random.seed(iter_seed)

            input_tensor = torch.randn((M, K), dtype=dtype, device='cuda').clone().contiguous()
            weight = torch.randn((K, N), dtype=dtype, device='cuda').clone().contiguous()

            # Layout随机化
            if config_dict.get('layout_type') != 'contiguous':
                weight = layout_randomizer.randomize_layout(weight)

            # Clone again before passing to operator (prevents pointer equality checks)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = operator(input_tensor.clone(), weight.clone())
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

            # Force memory release between iterations
            del input_tensor, weight

        return {
            'status': 'success',
            'times': times,
            'shape': config_dict['shape'],
            'seed': config_dict['seed'],
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'shape': config_dict['shape'],
            'seed': config_dict['seed'],
        }

    finally:
        sandbox.disable()
        cache_isolator.cleanup()


class CompetitionEvaluator:
    """竞赛评测主控制器"""

    def __init__(self,
                 operator_path: str,
                 config: CompetitionConfig = None):
        self.operator_path = operator_path
        self.config = config or CompetitionConfig()
        self.case_generator = TestCaseGenerator(self.config)
        from sandbox.timing_validator import StatisticalTimingValidator
        self.timing_validator = StatisticalTimingValidator(
            cv_threshold=self.config.cv_threshold,
            iqr_threshold=self.config.iqr_threshold
        )

    def run(self) -> Dict[str, Any]:
        """执行完整评测"""
        print("=" * 70)
        print("Triton 算子竞赛评测系统 - S级防护")
        print("=" * 70)

        # 生成测试用例
        test_cases = self.case_generator.generate_batch(self.config.num_tests)
        print(f"\n[INFO] 生成 {len(test_cases)} 组测试用例")

        # 执行评测
        results = []
        for i, case in enumerate(test_cases):
            print(f"\n[Test {i+1}/{len(test_cases)}]")
            print(f"  Shape: {case.shape}, dtype: {case.dtype}")

            result = self._run_single_test(case)
            results.append(result)

            if result['status'] == 'success':
                import numpy as np
                validation = self.timing_validator.validate(result['times'])
                status = "PASS" if validation.is_valid else "WARN"
                median = float(np.median(result['times'])) * 1000
                print(f"  Time: {median:.3f}ms [{status}]")
                if not validation.is_valid:
                    print(f"  Warning: {validation.message}")
            else:
                print(f"  ERROR: {result.get('error', 'Unknown error')}")

        # 复测校验
        retest_count = int(len(test_cases) * self.config.retest_ratio)
        if retest_count > 0:
            print(f"\n[INFO] 执行 {retest_count} 组复测校验...")
            retest_indices = random.sample(range(len(test_cases)), retest_count)
            for idx in retest_indices:
                case = test_cases[idx]
                retest_result = self._run_single_test(case)

                if results[idx]['status'] == 'success' and retest_result['status'] == 'success':
                    is_consistent, msg = self.timing_validator.retest_comparison(
                        results[idx]['times'],
                        retest_result['times']
                    )
                    if not is_consistent:
                        print(f"  [WARN] Case {idx+1} 复测不一致: {msg}")
                        results[idx]['retest_warning'] = msg

        # 汇总结果
        return self._summarize_results(results)

    def _run_single_test(self, case: TestCase) -> Dict[str, Any]:
        """执行单次测试（进程隔离）"""
        # Generate unique seed per iteration to prevent inter-iteration caching
        per_iteration_seeds = [
            case.seed + i * 10007 + random.randint(0, 99991)
            for i in range(self.config.timing_runs)
        ]
        config_dict = {
            'seed': case.seed,
            'shape': case.shape,
            'dtype': case.dtype,
            'layout_type': case.layout_type,
            'operator_path': self.operator_path,
            'warmup_runs': self.config.warmup_runs,
            'timing_runs': self.config.timing_runs,
            'per_iteration_seeds': per_iteration_seeds,
            'warmup_seed': case.seed - 1,
        }

        ctx = mp.get_context('spawn')
        with ctx.Pool(1) as pool:
            result = pool.apply(isolated_worker_main, (config_dict,))

        return result

    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """汇总结果"""
        import numpy as np
        success_results = [r for r in results if r['status'] == 'success']
        valid_results = []

        for r in success_results:
            validation = self.timing_validator.validate(r['times'])
            if validation.is_valid:
                valid_results.append(r)

        if not valid_results:
            return {
                'status': 'failed',
                'message': 'No valid test results',
                'results': results
            }

        times_list = [float(np.median(r['times'])) for r in valid_results]
        avg_time = float(np.mean(times_list))
        std_time = float(np.std(times_list))

        return {
            'status': 'success',
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(success_results),
                'valid_tests': len(valid_results),
                'average_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
            },
            'details': results
        }


# ============================================================
# 入口
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <operator.py>")
        sys.exit(1)

    operator_path = sys.argv[1]

    config = CompetitionConfig(
        num_tests=10,
        warmup_runs=1,
        timing_runs=4,
        retest_ratio=0.2,
    )

    evaluator = CompetitionEvaluator(operator_path, config)
    result = evaluator.run()

    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)

    if result['status'] == 'success':
        summary = result['summary']
        print(f"状态: 成功")
        print(f"平均时间: {summary['average_time_ms']:.3f} ms")
        print(f"标准差: {summary['std_time_ms']:.3f} ms")
        print(f"有效测试: {summary['valid_tests']}/{summary['total_tests']}")
    else:
        print(f"状态: 失败")
        print(f"原因: {result.get('message', 'Unknown')}")

    # 输出JSON结果
    print("\n[JSON OUTPUT]")
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
