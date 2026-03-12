#!/usr/bin/env python3
"""
Classify all 169 Kaldi K1 kernels into categories for batch processing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1
from collections import defaultdict
import json


def classify_kernel(name, info):
    """Classify a kernel based on its parameter signature."""
    input_args = info.get('input_args', [])
    
    if not input_args:
        return 'empty'
    
    # Check if it's a simple torch.Tensor interface (already done)
    if any(arg.get('type') == 'torch.Tensor' for arg in input_args):
        return 'simple_torch'
    
    # Check if it's a BLAS wrapper
    if any(arg.get('type') == 'cublasHandle_t' for arg in input_args):
        return 'cublas'
    
    # Check if it's a cudadecoder kernel
    if any(arg.get('type') in ['const DeviceParams&', 'const KernelParams&'] for arg in input_args):
        return 'cudadecoder'
    
    # Check if it's a cudafeat kernel
    if any(arg.get('type') in ['const LaneDesc*', 'const BatchedStaticNnet3KernelParams&'] for arg in input_args):
        return 'cudafeat'
    
    # Check parameter pattern
    first_arg_type = input_args[0].get('type', '')
    second_arg_type = input_args[1].get('type', '') if len(input_args) > 1 else ''
    
    # Standard pattern: dim3 Gr, dim3 Bl, ...
    if first_arg_type == 'dim3' and second_arg_type == 'dim3':
        # Check if it uses MatrixDim
        if any(arg.get('type') in ['MatrixDim', 'const MatrixDim'] for arg in input_args):
            return 'standard_dim3_matrixdim'
        else:
            return 'standard_dim3_other'
    
    # Vector pattern: int Gr, int Bl, ...
    if first_arg_type == 'int' and second_arg_type == 'int':
        arg_names = [arg.get('name', '') for arg in input_args[:2]]
        if 'Gr' in arg_names and 'Bl' in arg_names:
            return 'vector_int_grid'
    
    # Size_t pattern
    if first_arg_type == 'size_t' and second_arg_type == 'size_t':
        return 'sizet_grid'
    
    # Special matrix operations (no grid parameters)
    if first_arg_type in ['const double*', 'double*', 'int32_t', 'const BaseFloat*']:
        return 'no_grid_special'
    
    return 'other'


def analyze_kernels():
    """Analyze all kernels and classify them."""
    categories = defaultdict(list)
    
    for name, info in IMPL_INFO_K1.items():
        category = classify_kernel(name, info)
        categories[category].append(name)
    
    return categories


def print_analysis(categories):
    """Print detailed analysis of kernel categories."""
    print("=" * 80)
    print("KALDI K1 KERNEL CLASSIFICATION REPORT")
    print("=" * 80)
    print(f"\nTotal kernels: {sum(len(v) for v in categories.values())}\n")
    
    # Sort categories by count
    sorted_categories = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, kernels in sorted_categories:
        print(f"\n{category.upper()} ({len(kernels)} kernels)")
        print("-" * 80)
        for i, kernel in enumerate(kernels[:10], 1):  # Show first 10
            print(f"  {i:2d}. {kernel}")
        if len(kernels) > 10:
            print(f"  ... and {len(kernels) - 10} more")
    
    return sorted_categories


def get_priority_order(categories):
    """Define processing priority for each category."""
    priority_map = {
        'simple_torch': (1, 'Already completed ✓'),
        'standard_dim3_matrixdim': (2, 'Phase 1: Standard pattern - HIGH PRIORITY'),
        'vector_int_grid': (3, 'Phase 1: Vector pattern'),
        'sizet_grid': (4, 'Phase 1: Size_t pattern'),
        'standard_dim3_other': (5, 'Phase 1: Other dim3 pattern'),
        'cudadecoder': (6, 'Phase 2: Complex decoder kernels'),
        'cudafeat': (7, 'Phase 2: Feature extraction kernels'),
        'cublas': (8, 'Phase 3: BLAS wrappers'),
        'no_grid_special': (9, 'Phase 3: Special cases'),
        'other': (10, 'Phase 3: Other edge cases'),
        'empty': (11, 'Phase 3: Empty/invalid'),
    }
    
    result = []
    for category, kernels in categories:
        priority, description = priority_map.get(category, (99, 'Unknown'))
        result.append({
            'category': category,
            'priority': priority,
            'description': description,
            'count': len(kernels),
            'kernels': kernels
        })
    
    result.sort(key=lambda x: x['priority'])
    return result


def save_classification(priority_order, output_file):
    """Save classification to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(priority_order, f, indent=2)
    print(f"\n✓ Classification saved to: {output_file}")


def main():
    categories = analyze_kernels()
    sorted_categories = print_analysis(categories)
    
    print("\n" + "=" * 80)
    print("PROCESSING PRIORITY")
    print("=" * 80)
    
    priority_order = get_priority_order(sorted_categories)
    
    phase1_count = 0
    phase2_count = 0
    phase3_count = 0
    
    for item in priority_order:
        symbol = '✓' if item['category'] == 'simple_torch' else '→'
        print(f"\n{symbol} {item['description']}")
        print(f"  Category: {item['category']}")
        print(f"  Count: {item['count']} kernels")
        
        if 'Phase 1' in item['description']:
            phase1_count += item['count']
        elif 'Phase 2' in item['description']:
            phase2_count += item['count']
        elif 'Phase 3' in item['description']:
            phase3_count += item['count']
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 1 (Standard patterns): {phase1_count} kernels")
    print(f"Phase 2 (Special types):     {phase2_count} kernels")
    print(f"Phase 3 (Edge cases):        {phase3_count} kernels")
    print(f"Already completed:           3 kernels")
    print(f"Total:                       {phase1_count + phase2_count + phase3_count + 3} kernels")
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), '../../cache/kernel_classification.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_classification(priority_order, output_file)
    
    return priority_order


if __name__ == '__main__':
    main()
