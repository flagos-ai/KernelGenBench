#!/usr/bin/env python3
"""
批量生成 cuBLAS baseline 代码
使用金山云 API (glm-4.7)
"""
import os
import sys
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import logging

# 动态导入配置，避免触发 generator/__init__.py
import importlib.util

# 加载 config
config_path = Path(__file__).parent.parent / "cublas_c_api_config.py"
spec = importlib.util.spec_from_file_location("cublas_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
CUBLAS_C_API_CONFIG = config_module.CUBLAS_C_API_CONFIG

# 加载 prompt builder
builder_path = Path(__file__).parent / "baseline_prompt_builder.py"
spec2 = importlib.util.spec_from_file_location("prompt_builder", builder_path)
builder_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(builder_module)
BaselinePromptBuilder = builder_module.BaselinePromptBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KingsoftAPIClient:
    """金山云 API 客户端"""

    def __init__(self, api_key: str):
        self.base_url = "https://kspmas.ksyun.com/v1/chat/completions"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    async def generate(self, prompt: str, model: str = "glm-4.7",
                      temperature: float = 0.3, max_tokens: int = 16000) -> str:
        """调用 API 生成代码"""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens  # 使用 max_completion_tokens 而不是 max_tokens
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers,
                                   json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message'].get('content')
                    if not content:
                        # 尝试获取 reasoning_content
                        content = result['choices'][0]['message'].get('reasoning_content')
                    if not content:
                        raise Exception(f"API returned empty content: {result}")
                    return content
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")


class BaselineGenerator:
    """Baseline 代码生成器"""

    def __init__(self, api_key: str):
        self.api_client = KingsoftAPIClient(api_key)
        self.prompt_builder = BaselinePromptBuilder()

    async def generate_one(self, func_name: str, config: Dict[str, Any],
                          model: str = "glm-4.7") -> str:
        """生成单个 baseline 函数"""
        try:
            # 构建 prompt
            prompt = self.prompt_builder.build_prompt(func_name, config)

            # 调用 API
            code = await self.api_client.generate(prompt, model=model)

            # 提取代码块
            code = self._extract_code(code)

            return code

        except Exception as e:
            logger.error(f"Error generating {func_name}: {e}")
            return ""

    def _extract_code(self, text: str) -> str:
        """从 LLM 输出中提取代码块"""
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            code = text.split("```")[1].split("```")[0].strip()
        else:
            code = text.strip()
        return code

    async def generate_selected(self, func_list: List[str], output_dir: str,
                               concurrency: int = 10, model: str = "glm-4.7") -> List[Dict]:
        """生成指定的 baseline 函数"""
        from datetime import datetime

        logger.info("=" * 80)
        logger.info("批量生成 cuBLAS baseline 代码")
        logger.info("=" * 80)
        logger.info(f"并发数: {concurrency}")
        logger.info(f"模型: {model}")
        logger.info(f"总函数数: {len(func_list)}")

        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temperature = 0.3
        dir_name = f"baseline_cublas_c_api_{model}_temp_{temperature}_{timestamp}"
        output_path = Path(output_dir) / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录: {output_path}")

        # 准备任务
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(func_name: str, config: Dict[str, Any]):
            async with semaphore:
                code = await self.generate_one(func_name, config, model=model)
                return func_name, config, code

        # 创建任务（只生成指定的函数）
        tasks = [
            generate_with_semaphore(func_name, CUBLAS_C_API_CONFIG[func_name])
            for func_name in func_list
        ]

        return await self._execute_tasks(tasks, output_path)

    async def generate_all(self, output_dir: str, concurrency: int = 10,
                          model: str = "glm-4.7") -> List[Dict]:
        """批量生成所有 baseline"""
        return await self.generate_selected(
            func_list=list(CUBLAS_C_API_CONFIG.keys()),
            output_dir=output_dir,
            concurrency=concurrency,
            model=model
        )

    async def _execute_tasks(self, tasks, output_path: Path) -> List[Dict]:
        """执行生成任务并保存结果"""
        results = []
        success_count = 0
        fail_count = 0

        logger.info("开始生成 baseline 代码...")

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="生成进度"):
            func_name, config, code = await coro

            if code:
                # 为每个函数创建独立目录
                func_dir = output_path / func_name
                func_dir.mkdir(parents=True, exist_ok=True)

                # 保存代码文件
                code_file = func_dir / f"{func_name}.py"
                code_file.write_text(code)

                # 保存 prompt 文件
                prompt = self.prompt_builder.build_prompt(func_name, config)
                prompt_file = func_dir / "prompt.txt"
                prompt_file.write_text(prompt)

                results.append({
                    'func_name': func_name,
                    'func_dir': str(func_dir),
                    'success': True
                })
                success_count += 1
            else:
                results.append({
                    'func_name': func_name,
                    'success': False
                })
                fail_count += 1

        logger.info(f"生成完成: 成功 {success_count}, 失败 {fail_count}")
        return results


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="批量生成 cuBLAS baseline 代码")
    parser.add_argument("--output-dir",
                       default="/share/project/zpy/flagbench/src/generator/baseline",
                       help="输出基础目录（会在此目录下创建带时间戳的子目录）")
    parser.add_argument("--func-names", type=str,
                       help="指定要生成的函数名，多个用逗号分隔（如：cublasSaxpy_v2,cublasSdot_v2）")
    parser.add_argument("--all", action="store_true",
                       help="生成所有 284 个函数")
    parser.add_argument("--concurrency", type=int, default=10,
                       help="并发数")
    parser.add_argument("--model", default="glm-4.7",
                       help="使用的模型")
    parser.add_argument("--api-key",
                       default="8407460c-9a3d-4a32-bb0d-43e91a74304f",
                       help="金山云 API Key")

    args = parser.parse_args()

    # 检查参数
    if not args.all and not args.func_names:
        parser.error("必须指定 --func-names 或 --all")

    # 创建生成器
    generator = BaselineGenerator(args.api_key)

    # 确定要生成的函数列表
    if args.all:
        func_list = list(CUBLAS_C_API_CONFIG.keys())
        logger.info(f"生成所有 {len(func_list)} 个函数")
    else:
        func_list = [name.strip() for name in args.func_names.split(',')]
        # 验证函数名
        invalid_funcs = [f for f in func_list if f not in CUBLAS_C_API_CONFIG]
        if invalid_funcs:
            logger.error(f"无效的函数名: {invalid_funcs}")
            return
        logger.info(f"生成指定的 {len(func_list)} 个函数: {func_list}")

    # 批量生成
    results = await generator.generate_selected(
        func_list=func_list,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        model=args.model
    )

    # 保存结果摘要
    summary_path = Path("/tmp/baseline_generation_summary.json")
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info(f"结果摘要已保存到: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())

