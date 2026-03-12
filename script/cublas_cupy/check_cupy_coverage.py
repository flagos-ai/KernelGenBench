"""
检查CuPy cuBLAS API的可用性
为239个cuBLAS函数生成覆盖报告
"""
import json
import os

# 加载cuBLAS算子schema
schema_path = '/share/project/zpy/flagbench/script/cublas_cupy/cublas_ops.json'
with open(schema_path, 'r') as f:
    schema_data = json.load(f)

cublas_ops = schema_data.get('functions', [])
print(f"Total cuBLAS operations in schema: {len(cublas_ops)}")
print(f"Schema version: {schema_data.get('version', 'unknown')}")

# CuPy cuBLAS模块中已知可用的函数
# 基于CuPy官方文档: https://docs.cupy.dev/en/stable/reference/cublas.html
CUPY_CUBLAS_AVAILABLE = {
    # Level 1 BLAS (vector-vector operations)
    'asum': ['cublasSasum', 'cublasDasum', 'cublasScasum', 'cublasDzasum'],
    'axpy': ['cublasSaxpy', 'cublasDaxpy', 'cublasCaxpy', 'cublasZaxpy'],
    'copy': [],  # CuPy没有直接的cublas.copy包装
    'dot': ['cublasSdot', 'cublasDdot'],  # 实数版本
    'dotu': ['cublasCdotu', 'cublasZdotu'],  # 复数无共轭版本
    'dotc': ['cublasCdotc', 'cublasZdotc'],  # 复数共轭版本
    'nrm2': ['cublasSnrm2', 'cublasDnrm2', 'cublasScnrm2', 'cublasDznrm2'],
    'rot': [],  # 没有直接包装
    'rotg': [],
    'rotm': [],
    'rotmg': [],
    'scal': ['cublasSscal', 'cublasDscal', 'cublasCscal', 'cublasZscal', 'cublasCsscal', 'cublasZdscal'],
    'swap': [],  # 没有直接包装
    'iamax': ['cublasIsamax', 'cublasIdamax', 'cublasIcamax', 'cublasIzamax'],
    'iamin': ['cublasIsamin', 'cublasIdamin', 'cublasIcamin', 'cublasIzamin'],
    
    # Level 2 BLAS (matrix-vector operations)
    'gbmv': [],
    'gemv': ['cublasSgemv', 'cublasDgemv', 'cublasCgemv', 'cublasZgemv'],
    'ger': ['cublasSger', 'cublasDger'],  # 实数版本
    'geru': ['cublasCgeru', 'cublasZgeru'],  # 复数无共轭版本
    'gerc': ['cublasCgerc', 'cublasZgerc'],  # 复数共轭版本
    'sbmv': ['cublasSsbmv', 'cublasDsbmv'],
    'spmv': [],
    'spr': [],
    'spr2': [],
    'symv': [],
    'syr': [],
    'syr2': [],
    'tbmv': [],
    'tbsv': [],
    'tpmv': [],
    'tpsv': [],
    'trmv': [],
    'trsv': [],
    'hbmv': [],
    'hemv': [],
    'her': [],
    'her2': [],
    'hpmv': [],
    'hpr': [],
    'hpr2': [],
    
    # Level 3 BLAS (matrix-matrix operations)
    'gemm': ['cublasSgemm', 'cublasDgemm', 'cublasCgemm', 'cublasZgemm', 'cublasHgemm'],
    'symm': [],
    'syrk': ['cublasSsyrk', 'cublasDsyrk', 'cublasCsyrk', 'cublasZsyrk'],
    'syr2k': [],
    'trmm': [],
    'trsm': [],
    'hemm': [],
    'herk': [],
    'her2k': [],
    
    # BLAS-like extensions
    'geam': ['cublasSgeam', 'cublasDgeam', 'cublasCgeam', 'cublasZgeam'],
    'dgmm': ['cublasSdgmm', 'cublasDdgmm', 'cublasCdgmm', 'cublasZdgmm'],
    'gemmEx': [],
    'gemmBatched': [],
    'gemmStridedBatched': [],
}

# 从operation名称提取基础操作名（去除类型前缀）
def extract_base_operation(cublas_name):
    """
    从cublas函数名提取基础操作名
    例如: cublasSaxpy_v2 -> axpy, cublasDgemm_v2 -> gemm
    """
    # 移除 'cublas' 前缀
    if cublas_name.startswith('cublas'):
        name = cublas_name[6:]  # 去掉 'cublas'
    else:
        return None
    
    # 移除版本后缀 _v2
    if name.endswith('_v2'):
        name = name[:-3]
    
    # 移除类型前缀 (S/D/C/Z/H/I)
    if len(name) > 1:
        first_char = name[0]
        if first_char in ['S', 'D', 'C', 'Z', 'H', 'I']:
            base_op = name[1:].lower()
            return base_op
        # 处理特殊情况：Sc, Cs, Zd, Dz (混合精度scal)
        elif len(name) > 2 and name[:2] in ['Sc', 'Cs', 'Zd', 'Dz']:
            base_op = name[2:].lower()
            return base_op
    
    return name.lower()

# 分析每个cuBLAS函数
results = {
    'with_cupy': [],      # 有CuPy包装的
    'without_cupy': [],   # 没有CuPy包装的
    'unknown': []         # 无法确定的
}

operation_summary = {}  # 按操作分组统计

for op in cublas_ops:
    func_name = op.get('name', '')
    base_op = extract_base_operation(func_name)
    
    if base_op is None:
        results['unknown'].append(op)
        continue
    
    # 检查该函数是否在CuPy的已知列表中
    has_cupy = False
    if base_op in CUPY_CUBLAS_AVAILABLE:
        cupy_funcs = CUPY_CUBLAS_AVAILABLE[base_op]
        # 检查具体的函数名（去掉_v2后缀）
        func_name_without_v2 = func_name.replace('_v2', '')
        if func_name_without_v2 in cupy_funcs:
            has_cupy = True
            results['with_cupy'].append(op)
        else:
            results['without_cupy'].append(op)
    else:
        results['without_cupy'].append(op)
    
    # 统计
    if base_op not in operation_summary:
        operation_summary[base_op] = {
            'total': 0,
            'with_cupy': 0,
            'without_cupy': 0,
            'functions': []
        }
    
    operation_summary[base_op]['total'] += 1
    operation_summary[base_op]['functions'].append({
        'name': func_name,
        'has_cupy': has_cupy,
        'dtype': op.get('dtype', 'unknown')
    })
    
    if has_cupy:
        operation_summary[base_op]['with_cupy'] += 1
    else:
        operation_summary[base_op]['without_cupy'] += 1

# 打印统计信息
print("\n" + "="*80)
print("CuPy cuBLAS API Coverage Analysis")
print("="*80)
print(f"\nTotal cuBLAS functions: {len(cublas_ops)}")
print(f"With CuPy wrapper: {len(results['with_cupy'])}")
print(f"Without CuPy wrapper: {len(results['without_cupy'])}")
print(f"Unknown: {len(results['unknown'])}")
print(f"Coverage: {len(results['with_cupy'])/len(cublas_ops)*100:.1f}%")

# 生成Markdown文档
md_content = f"""# CuPy cuBLAS API Coverage Report

**生成时间**: {os.popen('date').read().strip()}

## 总览

- **总cuBLAS函数数**: {len(cublas_ops)}
- **有CuPy包装**: {len(results['with_cupy'])} ({len(results['with_cupy'])/len(cublas_ops)*100:.1f}%)
- **无CuPy包装**: {len(results['without_cupy'])} ({len(results['without_cupy'])/len(cublas_ops)*100:.1f}%)
- **无法确定**: {len(results['unknown'])}

## 按操作类型分类

"""

# 按BLAS级别分组
level1_ops = ['asum', 'axpy', 'copy', 'dot', 'dotu', 'dotc', 'nrm2', 'rot', 'rotg', 'rotm', 'rotmg', 'scal', 'swap', 'iamax', 'iamin']
level2_ops = ['gbmv', 'gemv', 'ger', 'geru', 'gerc', 'sbmv', 'spmv', 'spr', 'spr2', 'symv', 'syr', 'syr2', 'tbmv', 'tbsv', 'tpmv', 'tpsv', 'trmv', 'trsv', 'hbmv', 'hemv', 'her', 'her2', 'hpmv', 'hpr', 'hpr2']
level3_ops = ['gemm', 'symm', 'syrk', 'syr2k', 'trmm', 'trsm', 'hemm', 'herk', 'her2k']
extension_ops = ['geam', 'dgmm', 'gemmEx', 'gemmBatched', 'gemmStridedBatched']

for level_name, ops_list in [
    ('Level 1 BLAS (Vector-Vector)', level1_ops),
    ('Level 2 BLAS (Matrix-Vector)', level2_ops),
    ('Level 3 BLAS (Matrix-Matrix)', level3_ops),
    ('BLAS-like Extensions', extension_ops)
]:
    md_content += f"\n### {level_name}\n\n"
    md_content += "| Operation | Total | With CuPy | Without CuPy | Coverage |\n"
    md_content += "|-----------|-------|-----------|--------------|----------|\n"
    
    for op in ops_list:
        if op in operation_summary:
            stats = operation_summary[op]
            coverage = stats['with_cupy'] / stats['total'] * 100 if stats['total'] > 0 else 0
            status = "✅" if coverage == 100 else "⚠️" if coverage > 0 else "❌"
            md_content += f"| {status} **{op}** | {stats['total']} | {stats['with_cupy']} | {stats['without_cupy']} | {coverage:.0f}% |\n"

# 详细列表
md_content += "\n## 详细函数列表\n\n"

# 有CuPy包装的函数
md_content += "### ✅ 有CuPy cuBLAS包装的函数 (推荐直接调用)\n\n"
md_content += "| cuBLAS Function | Operation | Data Type | CuPy API |\n"
md_content += "|----------------|-----------|-----------|----------|\n"

for op in sorted(results['with_cupy'], key=lambda x: x['name']):
    func_name = op['name']
    base_op = extract_base_operation(func_name)
    dtype = op.get('dtype', 'unknown')
    md_content += f"| `{func_name}` | {base_op} | {dtype} | `cupy.cublas.{base_op}()` |\n"

# 没有CuPy包装的函数
md_content += f"\n### ❌ 没有CuPy cuBLAS包装的函数 ({len(results['without_cupy'])}个，需要手动实现)\n\n"
md_content += "| cuBLAS Function | Operation | Data Type | 建议实现方式 |\n"
md_content += "|----------------|-----------|-----------|------------|\n"

for op in sorted(results['without_cupy'], key=lambda x: x['name']):
    func_name = op['name']
    base_op = extract_base_operation(func_name)
    dtype = op.get('dtype', 'unknown')
    md_content += f"| `{func_name}` | {base_op} | {dtype} | CuPy数组操作 |\n"

# 使用建议
md_content += """
## 代码生成建议

### 方案1：有CuPy包装的算子（推荐直接调用）

对于上表中"✅ 有CuPy cuBLAS包装"的函数，生成时应直接调用 CuPy cuBLAS API：

```python
from cupy import cublas

# 例如：axpy操作
result = cublas.axpy(alpha, x, y)  # 直接调用，无需create()
```

**注意事项**：
- ❌ 不要使用 `cublas.create()` - CuPy会自动管理handle
- ✅ 直接调用函数，CuPy内部处理所有细节
- ✅ 参数顺序遵循CuPy文档，而非原始cuBLAS C API

### 方案2：没有CuPy包装的算子（手动实现）

对于没有CuPy包装的函数，使用CuPy数组操作手动实现：

```python
import cupy as cp

# 例如：copy操作（假设没有直接包装）
y_cp = x_cp.copy()  # 使用CuPy数组方法
```

## 生成优先级

建议按以下顺序生成baseline：

1. **高优先级**：Level 1 BLAS 有CuPy包装的（{sum(1 for op in level1_ops if op in operation_summary and operation_summary[op]['with_cupy'] > 0)} 个操作）
2. **高优先级**：Level 3 BLAS 有CuPy包装的（gemm, syrk等核心矩阵操作）
3. **中优先级**：Level 2 BLAS 有CuPy包装的
4. **低优先级**：需要手动实现的操作

## CuPy cuBLAS API参考

CuPy官方文档：https://docs.cupy.dev/en/stable/reference/cublas.html

### 常用函数签名

```python
# Level 1
cupy.cublas.axpy(alpha, x, y)  # y = alpha*x + y
cupy.cublas.dot(x, y)          # 返回 x.T @ y
cupy.cublas.nrm2(x)            # 返回 ||x||_2
cupy.cublas.scal(alpha, x)     # x = alpha*x

# Level 2  
cupy.cublas.gemv(trans, A, x, alpha=1.0, beta=0.0, y=None)  # y = alpha*op(A)@x + beta*y

# Level 3
cupy.cublas.gemm(transa, transb, A, B, C=None, alpha=1.0, beta=0.0)  # C = alpha*op(A)@op(B) + beta*C
```
"""

# 保存Markdown文件
output_path = '/share/project/zpy/flagbench/script/cublas_cupy/cupy_cublas_coverage.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n✓ Markdown报告已保存到: {output_path}")

# 保存JSON格式的详细数据
json_output = {
    'summary': {
        'total': len(cublas_ops),
        'with_cupy': len(results['with_cupy']),
        'without_cupy': len(results['without_cupy']),
        'unknown': len(results['unknown']),
        'coverage_percentage': len(results['with_cupy'])/len(cublas_ops)*100
    },
    'operations': operation_summary,
    'functions': {
        'with_cupy': [op['name'] for op in results['with_cupy']],
        'without_cupy': [op['name'] for op in results['without_cupy']],
        'unknown': [op['name'] for op in results['unknown']]
    }
}

json_output_path = '/share/project/zpy/flagbench/script/cublas_cupy/cupy_cublas_coverage.json'
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, indent=2)

print(f"✓ JSON数据已保存到: {json_output_path}")

# 打印一些关键统计
print("\n" + "="*80)
print("关键操作的CuPy支持情况")
print("="*80)

priority_ops = ['gemm', 'axpy', 'dot', 'gemv', 'scal', 'syrk', 'geam']
for op in priority_ops:
    if op in operation_summary:
        stats = operation_summary[op]
        print(f"{op:10s}: {stats['with_cupy']}/{stats['total']} 有CuPy包装 ({stats['with_cupy']/stats['total']*100:.0f}%)")
