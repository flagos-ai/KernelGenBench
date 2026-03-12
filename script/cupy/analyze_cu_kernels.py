#!/usr/bin/env python3
"""
分析 cu-kernels.cu 文件，提取所有 __global__ kernel 的签名和源码
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

CU_KERNELS_PATH = "/share/project/zpy/k1_repo/src/cudamatrix/cu-kernels.cu"


def extract_kernel_signatures(cu_file: Path) -> List[Dict]:
    """
    从 .cu 文件中提取所有 __global__ kernel 的签名
    
    Returns:
        List of dicts with: name, return_type, params, full_signature, start_line
    """
    with open(cu_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        lines = content.split('\n')
    
    kernels = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 查找 __global__ 声明
        if '__global__' in line:
            # 收集完整的函数签名（可能跨多行）
            signature_lines = []
            j = i
            
            # 向后查找函数签名的开始
            while j < len(lines):
                signature_lines.append(lines[j])
                # 找到函数体开始的 '{'
                if '{' in lines[j]:
                    break
                j += 1
            
            # 合并多行签名
            full_signature = ' '.join(signature_lines)
            full_signature = re.sub(r'\s+', ' ', full_signature).strip()
            
            # 提取函数名
            # 匹配模式: __global__ ... kernel_name(...)
            match = re.search(r'__global__\s+(?:static\s+)?(?:inline\s+)?(?:\w+\s+)?(\w+)\s*\(', full_signature)
            
            if match:
                kernel_name = match.group(1)
                
                # 提取参数列表
                params_match = re.search(r'\((.+?)\)\s*\{', full_signature)
                params_str = params_match.group(1) if params_match else ""
                
                # 提取返回类型
                return_type_match = re.search(r'__global__\s+(?:static\s+)?(?:inline\s+)?([\w\s\*]+?)\s+' + kernel_name, full_signature)
                return_type = return_type_match.group(1).strip() if return_type_match else "void"
                
                kernels.append({
                    'name': kernel_name,
                    'return_type': return_type,
                    'params': params_str,
                    'full_signature': full_signature[:200],  # 截取前200字符
                    'start_line': i + 1,
                })
            
            i = j
        
        i += 1
    
    return kernels


def extract_kernel_source(cu_file: Path, kernel_name: str, start_line: int) -> str:
    """
    从 .cu 文件中提取指定 kernel 的完整源码
    
    Args:
        cu_file: .cu 文件路径
        kernel_name: kernel 名称
        start_line: 函数开始行号（1-based）
    
    Returns:
        完整的 kernel 源码
    """
    with open(cu_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # 从 start_line 开始查找
    i = start_line - 1
    source_lines = []
    brace_count = 0
    started = False
    
    while i < len(lines):
        line = lines[i]
        source_lines.append(line)
        
        # 统计大括号
        for char in line:
            if char == '{':
                brace_count += 1
                started = True
            elif char == '}':
                brace_count -= 1
        
        # 找到匹配的闭括号
        if started and brace_count == 0:
            break
        
        i += 1
    
    return ''.join(source_lines)


def parse_parameters(params_str: str) -> List[Dict]:
    """
    解析参数字符串为结构化数据
    
    Args:
        params_str: "int a, float* b, const T c"
    
    Returns:
        [{"type": "int", "name": "a"}, {"type": "float*", "name": "b"}, ...]
    """
    if not params_str.strip():
        return []
    
    params = []
    # 分割参数（简单版本，不处理嵌套）
    for param in params_str.split(','):
        param = param.strip()
        if not param:
            continue
        
        # 提取类型和名称
        # 模式: "type name" or "type* name" or "const type name"
        parts = param.split()
        if len(parts) >= 2:
            param_name = parts[-1]
            param_type = ' '.join(parts[:-1])
            params.append({
                'type': param_type,
                'name': param_name
            })
    
    return params


def main():
    print("=" * 70)
    print("分析 cu-kernels.cu 文件")
    print("=" * 70)
    
    cu_file = Path(CU_KERNELS_PATH)
    
    if not cu_file.exists():
        print(f"错误: 文件不存在: {cu_file}")
        return
    
    print(f"\n文件: {cu_file}")
    print(f"大小: {cu_file.stat().st_size / 1024:.1f} KB")
    
    # 提取所有 kernel 签名
    print("\n提取 kernel 签名...")
    kernels = extract_kernel_signatures(cu_file)
    
    print(f"\n找到 {len(kernels)} 个 __global__ kernels\n")
    
    # 显示前 20 个
    print("前 20 个 kernels:")
    print("-" * 70)
    for i, kernel in enumerate(kernels[:20], 1):
        print(f"{i:3d}. {kernel['name']:40s} (行 {kernel['start_line']:5d})")
    
    if len(kernels) > 20:
        print(f"     ... 还有 {len(kernels) - 20} 个")
    
    # 保存到 JSON
    output_file = Path(__file__).parent / "cu_kernels_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(kernels, f, indent=2)
    
    print(f"\n完整列表已保存到: {output_file}")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("统计信息:")
    print("=" * 70)
    print(f"总计 kernel 数量: {len(kernels)}")
    
    # 检查与 IMPL_INFO_K1 的匹配
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1
    
    kernel_names = {k['name'] for k in kernels}
    impl_names = set(IMPL_INFO_K1.keys())
    
    matched = kernel_names & impl_names
    only_in_cu = kernel_names - impl_names
    only_in_impl = impl_names - kernel_names
    
    print(f"\nIMPL_INFO_K1 中定义: {len(impl_names)} 个")
    print(f"cu-kernels.cu 中找到: {len(kernel_names)} 个")
    print(f"匹配: {len(matched)} 个")
    print(f"仅在 .cu 中: {len(only_in_cu)} 个")
    print(f"仅在 IMPL_INFO_K1 中: {len(only_in_impl)} 个")
    
    if only_in_cu:
        print(f"\n仅在 .cu 中的前 10 个: {list(only_in_cu)[:10]}")
    
    if only_in_impl:
        print(f"\n仅在 IMPL_INFO_K1 中的前 10 个: {list(only_in_impl)[:10]}")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    # 示例：提取第一个 kernel 的完整源码
    if kernels:
        first_kernel = kernels[0]
        print(f"\n示例：提取 '{first_kernel['name']}' 的源码...")
        source = extract_kernel_source(cu_file, first_kernel['name'], first_kernel['start_line'])
        print(f"源码长度: {len(source)} 字符")
        print(f"前 500 字符:\n{source[:500]}")


if __name__ == "__main__":
    main()
