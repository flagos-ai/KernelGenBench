#!/usr/bin/env python3
"""检查注册了但没有测试的算子"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flagbench.dataset.kernel_list import IMPL_INFO, PYTORCH_OPERATORS

# 测试文件目录
TEST_DIR = Path(__file__).parent / "src" / "flagbench" / "accuracy"

# 测试文件列表
TEST_FILES = [
    "test_unary_pointwise_ops.py",
    "test_binary_pointwise_ops.py",
    "test_blas_ops.py",
    "test_special_ops.py",
    "test_norm_ops.py",
    "test_general_reduction_ops.py",
    "test_tensor_constructor_ops.py",
    "test_reduction_ops.py",
    "test_attention_ops.py",
    "test_distribution_ops.py",  # 包含 normal, uniform_, exponential_ 等
]


def extract_labels_from_file(filepath):
    """从测试文件中提取所有 @label 装饰器中的算子名称"""
    if not filepath.exists():
        return set()
    
    labels = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 @label("operator_name") 或 @label('operator_name')
    pattern = r'@label\(["\']([^"\']+)["\']\)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        # 跳过 "inplace" 标签
        if match != "inplace":
            labels.add(match)
    
    return labels


def get_all_test_labels():
    """收集所有测试文件中的 label"""
    all_labels = set()
    label_to_files = defaultdict(list)
    
    for filename in TEST_FILES:
        filepath = TEST_DIR / filename
        labels = extract_labels_from_file(filepath)
        all_labels.update(labels)
        
        for label in labels:
            label_to_files[label].append(filename)
    
    return all_labels, label_to_files


def get_registered_operators():
    """获取所有注册的算子名称"""
    # 从 IMPL_INFO 获取所有算子名称
    registered = set(IMPL_INFO.keys())
    
    # 注意：不要添加别名，因为别名在 IMPL_INFO 中已经存在
    # 别名（dropout, group_norm, layer_norm, weight_norm）已经在 IMPL_INFO 中
    # 它们通过 IMPL_INFO["dropout"] = IMPL_INFO["native_dropout"] 的方式映射
    
    return registered


def get_label_to_registered_mapping():
    """获取测试 label 到注册算子名称的映射"""
    # 这些是测试中使用的 label 名称，但注册时使用的是不同的名称
    label_to_registered = {
        # 别名映射（测试 label -> 注册算子名称）
        "dropout": "native_dropout",  # 测试中的 dropout label 对应注册的 native_dropout
        "native_dropout": "native_dropout",
        "group_norm": "native_group_norm",  # 测试中的 group_norm label 对应注册的 native_group_norm
        "native_group_norm": "native_group_norm",
        "layer_norm": "native_layer_norm",  # 测试中的 layer_norm label 对应注册的 native_layer_norm
        "native_layer_norm": "native_layer_norm",
        "weight_norm": "_weight_norm_interface",
        "weight_norm_interface": "_weight_norm_interface",
        # 其他映射
        "vector_norm": "linalg_vector_norm",
        # pad 和 constant_pad_nd 是两个不同的算子，pad 在注册列表中，不需要映射
        "upsample_bicubic2d_aa": "_upsample_bicubic2d_aa",
        "upsample": "_upsample_bicubic2d_aa",  # upsample 是别名
        "unique": "unique",  # unique 在 IMPL_INFO 中，注册的是 _unique2，但 key 是 unique
        "native_instance_norm": "instance_norm",
        "skip_rms_norm": "skip_layer_norm",  # 可能是别名
        "diagonal": "diagonal",  # diagonal 本身就在注册列表中
        "diagonal_backward": "diagonal",  # diagonal_backward 测试可能覆盖 diagonal，但注册的是 diagonal
        # 除法相关映射
        # 注意：div 和 divide 是不同的算子，都在 IMPL_INFO 中
        # div 和 div_ 有对应的测试 label，应该直接匹配
        "trunc_divide": "divide",  # trunc_divide 测试覆盖 divide
        "trunc_divide_": "divide_",  # trunc_divide_ 测试覆盖 divide_
        # true_divide 和 divide 是同一个功能，测试用 div 覆盖
        "true_divide": "divide",  # true_divide 和 divide 功能相同
        "true_divide_": "divide_",  # true_divide_ 和 divide_ 功能相同
        # 这些是测试中使用的，但可能不是标准算子
        "linear": None,  # 可能是 addmm/addmv 的别名
        "matmul": None,  # 可能是 mm/bmm 的别名
        "topk": "topk",  # topk 有测试
        "conv1d": "conv1d",  # conv1d 有测试
        "index_fill": "index_fill",  # index_fill 有测试
        "upsample_bicubic2d_aa": "_upsample_bicubic2d_aa",  # upsample 相关
        "upsample_nearest2d": "upsample_nearest2d",  # upsample 相关
        # upsample 在 IMPL_INFO 中映射到多个实现（_upsample_bicubic2d_aa 和 upsample_nearest2d）
        # 测试中分别用不同的 label，所以 upsample 本身不需要单独映射
        # 但需要确保 _upsample_bicubic2d_aa 和 upsample_nearest2d 都有测试
        "avg_pool2d": None,  # 可能没有注册
        "flash_mla": None,  # 可能是 flash_attention_forward 的别名
        "rsub": "sub",  # 可能是 sub 的别名
        "or_": "bitwise_or",  # 可能是 bitwise_or 的别名
        "or__": "bitwise_or_",  # 可能是 bitwise_or_ 的别名
    }
    return label_to_registered


def main():
    print("=" * 80)
    print("检查注册了但没有测试的算子")
    print("=" * 80)
    
    # 获取注册的算子
    registered_ops = get_registered_operators()
    print(f"\n✅ 注册的算子总数: {len(registered_ops)}")
    
    # 获取所有测试中的 label
    test_labels, label_to_files = get_all_test_labels()
    print(f"✅ 测试中的 label 总数: {len(test_labels)}")
    
    # 获取 label 到注册算子的映射
    label_mapping = get_label_to_registered_mapping()
    
    # 将测试 label 映射到注册的算子名称
    mapped_test_labels = set()
    unmapped_labels = set()
    
    for label in test_labels:
        if label in label_mapping:
            mapped_name = label_mapping[label]
            if mapped_name is not None:
                mapped_test_labels.add(mapped_name)
            else:
                unmapped_labels.add(label)  # 这些 label 没有对应的注册算子
        else:
            # 如果 label 本身就在注册列表中，直接使用
            if label in registered_ops:
                mapped_test_labels.add(label)
            else:
                unmapped_labels.add(label)
    
    # 特殊处理：别名（dropout, group_norm, layer_norm, weight_norm）在 IMPL_INFO 中已存在
    # 如果测试中有对应的 label，应该认为它们有测试
    aliases = ["dropout", "group_norm", "layer_norm", "weight_norm"]
    for alias in aliases:
        if alias in registered_ops and alias in test_labels:
            mapped_test_labels.add(alias)
    
    # 特殊处理：upsample 映射到多个实现，如果两个实现都有测试，则认为 upsample 有测试
    if "_upsample_bicubic2d_aa" in mapped_test_labels and "upsample_nearest2d" in mapped_test_labels:
        mapped_test_labels.add("upsample")
    
    # 特殊处理：true_divide 和 divide 功能相同，如果 divide 有测试，则认为 true_divide 也有测试
    if "divide" in mapped_test_labels:
        mapped_test_labels.add("true_divide")
    if "divide_" in mapped_test_labels:
        mapped_test_labels.add("true_divide_")
    
    # 找出注册了但没有测试的算子
    missing_tests = registered_ops - mapped_test_labels
    
    # 找出测试了但没有注册的算子（可能是别名或错误）
    extra_tests = unmapped_labels
    
    print("\n" + "=" * 80)
    print("📊 分析结果")
    print("=" * 80)
    
    if missing_tests:
        print(f"\n❌ 注册了但没有测试的算子 ({len(missing_tests)} 个):")
        print("-" * 80)
        for op in sorted(missing_tests):
            # 检查是否是自定义算子（通常不需要测试）
            custom_ops = {
                "apply_rotary_pos_emb", "concat_and_cache_mla", "conv_depthwise2d",
                "flash_attention_forward", "flash_attn_varlen_func", "fused_add_rms_norm",
                "get_scheduler_metadata", "reshape_and_cache", "reshape_and_cache_flash",
                "rwkv_ka_fusion", "rwkv_mm_sparsity", "topk_softmax", "weight_norm",
                "gelu_and_mul", "silu_and_mul", "skip_layer_norm",
            }
            
            if op in custom_ops:
                print(f"  ⚠️  {op:30s} (自定义算子，可能不需要测试)")
            else:
                print(f"  ❌ {op:30s}")
    else:
        print("\n✅ 所有注册的算子都有对应的测试！")
    
    if extra_tests:
        print(f"\n⚠️  测试了但没有注册的算子 ({len(extra_tests)} 个):")
        print("-" * 80)
        for op in sorted(extra_tests):
            files = label_to_files[op]
            print(f"  ⚠️  {op:30s} (出现在: {', '.join(files)})")
    
    # 详细统计
    print("\n" + "=" * 80)
    print("📈 详细统计")
    print("=" * 80)
    
    # 按文件统计
    print("\n按测试文件统计 label 数量:")
    for filename in TEST_FILES:
        filepath = TEST_DIR / filename
        labels = extract_labels_from_file(filepath)
        print(f"  {filename:40s}: {len(labels):3d} 个 label")
    
    # 显示一些常见的缺失算子
    if missing_tests:
        print("\n" + "=" * 80)
        print("💡 建议")
        print("=" * 80)
        print("以下算子已注册但缺少测试，建议添加测试:")
        
        # 过滤掉自定义算子
        standard_missing = [op for op in sorted(missing_tests) 
                          if op not in {
                              "apply_rotary_pos_emb", "concat_and_cache_mla", "conv_depthwise2d",
                              "flash_attention_forward", "flash_attn_varlen_func", "fused_add_rms_norm",
                              "get_scheduler_metadata", "reshape_and_cache", "reshape_and_cache_flash",
                              "rwkv_ka_fusion", "rwkv_mm_sparsity", "topk_softmax", "weight_norm",
                              "gelu_and_mul", "silu_and_mul", "skip_layer_norm",
                          }]
        
        if standard_missing:
            print("\n标准 PyTorch 算子:")
            for op in standard_missing[:20]:  # 只显示前20个
                print(f"  - {op}")
            if len(standard_missing) > 20:
                print(f"  ... 还有 {len(standard_missing) - 20} 个")


if __name__ == "__main__":
    main()

