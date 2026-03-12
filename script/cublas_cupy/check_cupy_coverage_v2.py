"""
重新检查CuPy cuBLAS API的可用性 - 用实际导入验证
"""
import json
import os

# 加载cuBLAS算子schema
schema_path = '/share/project/zpy/flagbench/script/cublas_cupy/cublas_ops.json'
with open(schema_path, 'r') as f:
    schema_data = json.load(f)

cublas_ops = schema_data.get('functions', [])
print(f"Total cuBLAS operations in schema: {len(cublas_ops)}")

# 实际测试CuPy中哪些函数可用
try:
    import cupy as cp
    from cupy import cublas
    
    # 获取实际可用的函数
    available_funcs = [x for x in dir(cublas) if not x.startswith("_") and callable(getattr(cublas, x, None))]
    print(f"CuPy实际可用的cuBLAS函数: {len(available_funcs)}")
    print(f"函数: {', '.join(sorted(available_funcs))}")
    
    # 特殊函数：gemm, gemv等可能需要检查签名
    print("\n检查关键函数的签名...")
    
    key_funcs = ['axpy', 'dot', 'gemm', 'gemv', 'scal', 'syrk', 'geam', 'dgmm']
    for fn in key_funcs:
        if hasattr(cublas, fn):
            import inspect
            try:
                sig = inspect.signature(getattr(cublas, fn))
                print(f"  ✓ cublas.{fn}{sig}")
            except:
                print(f"  ✓ cublas.{fn} (签名检查失败)")
        else:
            print(f"  ✗ cublas.{fn} 不存在")
            
except ImportError as e:
    print(f"CuPy不可用: {e}")
    available_funcs = []

# 从operation名称提取基础操作名
def extract_base_operation(cublas_name):
    if cublas_name.startswith('cublas'):
        name = cublas_name[6:]
    else:
        return None
    
    # 移除版本后缀 _v2, _v2_ex 等
    if name.endswith('_v2'):
        name = name[:-3]
    elif name.endswith('_ex'):
        name = name[:-3]
    elif name.endswith('_v2_ex'):
        name = name[:-5]
    
    # 移除类型前缀 (S/D/C/Z/H/I)
    if len(name) > 1:
        first_char = name[0]
        if first_char in ['S', 'D', 'C', 'Z', 'H', 'I']:
            base_op = name[1:].lower()
            return base_op
        elif len(name) > 2 and name[:2] in ['Sc', 'Cs', 'Zd', 'Dz']:
            base_op = name[2:].lower()
            return base_op
    
    return name.lower()

# 分析每个cuBLAS函数
results = {
    'with_cupy': [],
    'without_cupy': [],
    'unknown': []
}

for op in cublas_ops:
    func_name = op.get('name', '')
    base_op = extract_base_operation(func_name)
    
    if base_op is None:
        results['unknown'].append(op)
        continue
    
    # 检查是否在CuPy可用函数列表中
    if base_op in available_funcs:
        results['with_cupy'].append(op)
    else:
        results['without_cupy'].append(op)

# 打印统计
print("\n" + "="*80)
print("CuPy cuBLAS API Coverage Analysis (实际测试结果)")
print("="*80)
print(f"\nTotal cuBLAS functions: {len(cublas_ops)}")
print(f"With CuPy wrapper: {len(results['with_cupy'])}")
print(f"Without CuPy wrapper: {len(results['without_cupy'])}")
print(f"Coverage: {len(results['with_cupy'])/len(cublas_ops)*100:.1f}%")

# 按基础操作分组统计
operation_summary = {}
for op in cublas_ops:
    func_name = op.get('name', '')
    base_op = extract_base_operation(func_name)
    
    if base_op is None:
        continue
    
    if base_op not in operation_summary:
        operation_summary[base_op] = {'total': 0, 'with_cupy': 0, 'without_cupy': 0}
    
    operation_summary[base_op]['total'] += 1
    if base_op in available_funcs:
        operation_summary[base_op]['with_cupy'] += 1
    else:
        operation_summary[base_op]['without_cupy'] += 1

# 打印各操作的覆盖情况
print("\n按操作的覆盖情况:")
print("-" * 50)
for op in sorted(operation_summary.keys()):
    stats = operation_summary[op]
    coverage = stats['with_cupy'] / stats['total'] * 100 if stats['total'] > 0 else 0
    status = "✅" if coverage == 100 else "⚠️" if coverage > 0 else "❌"
    print(f"{status} {op:20s}: {stats['with_cupy']:2d}/{stats['total']:2d} ({coverage:5.1f}%)")

# 生成Markdown报告
md_content = f"""# CuPy cuBLAS API Coverage Report (实际测试)

**生成时间**: {os.popen('date').read().strip()}

## 总览

- **总cuBLAS函数数**: {len(cublas_ops)}
- **有CuPy包装**: {len(results['with_cupy'])} ({len(results['with_cupy'])/len(cublas_ops)*100:.1f}%)
- **无CuPy包装**: {len(results['without_cupy'])} ({len(results['without_cupy'])/len(cublas_ops)*100:.1f}%)
- **CuPy实际支持的函数**: {len(available_funcs)} 个

## CuPy实际支持的函数

```
{', '.join(sorted(available_funcs))}
```

## 按操作分类

| 操作 | 总数 | 有CuPy | 无CuPy | 覆盖率 | 状态 |
|------|------|--------|--------|--------|------|
"""

for op in sorted(operation_summary.keys()):
    stats = operation_summary[op]
    coverage = stats['with_cupy'] / stats['total'] * 100 if stats['total'] > 0 else 0
    status = "✅" if coverage == 100 else "⚠️" if coverage > 0 else "❌"
    md_content += f"| {op:15s} | {stats['total']:3d} | {stats['with_cupy']:3d} | {stats['without_cupy']:3d} | {coverage:5.1f}% | {status} |\n"

# 有CuPy包装的详细列表
md_content += """
## 有CuPy包装的函数

这些函数可以直接调用 `cupy.cublas.<function_name>()`：

| cuBLAS Function | Operation | Data Type |
|----------------|-----------|-----------|
"""

for op in sorted(results['with_cupy'], key=lambda x: x['name']):
    func_name = op['name']
    base_op = extract_base_operation(func_name)
    dtype = op.get('dtype', 'unknown')
    md_content += f"| `{func_name}` | {base_op} | {dtype} |\n"

# 没有CuPy包装的函数（按操作分组）
md_content += f"""
## 无CuPy包装的函数 ({len(results['without_cupy'])}个)

这些函数需要使用CuPy数组操作手动实现。

### 按操作分组

"""

for op in sorted(operation_summary.keys()):
    stats = operation_summary[op]
    if stats['without_cupy'] > 0:
        md_content += f"#### {op} ({stats['without_cupy']}个变体)\n\n"
        for func in results['without_cupy']:
            if extract_base_operation(func['name']) == op:
                md_content += f"- `{func['name']}`\n"
        md_content += "\n"

# 代码示例
md_content += """
## 代码生成建议

### 有CuPy包装的操作（直接调用）

```python
from cupy import cublas

# Level 1 BLAS
cublas.axpy(alpha, x, y)           # y = alpha*x + y
cublas.dot(x, y)                   # x.T @ y
cublas.scal(alpha, x)              # x = alpha*x
cublas.nrm2(x)                     # ||x||_2
cublas.asum(x)                     # sum(|x_i|)

# Level 2 BLAS
cublas.gemv(trans, A, x)           # y = alpha*op(A)@x + beta*y
cublas.ger(A, x, y)                # A = alpha*x@y.T + A
cublas.geru(A, x, y)               # A = alpha*x@y.T + A (unconjugated)
cublas.gerc(A, x, y)               # A = alpha*conj(x)@y.T + A

# Level 3 BLAS
cublas.gemm(transa, transb, A, B)  # C = alpha*op(A)@op(B) + beta*C
cublas.syrk(uplo, trans, A)        # C = alpha*A@A.T + beta*C
```

### 无CuPy包装的操作（手动实现）

对于没有CuPy包装的操作，使用CuPy数组操作：

```python
import cupy as cp

# 例如：copy操作
y = x.copy()

# 例如：swap操作  
y[:], x[:] = x.copy(), y.copy()

# 例如：rot操作（Givens旋转）
r = cp.sqrt(x**2 + y**2)
cos_val = x / r
sin_val = y / r
x_new = cos_val * x + sin_val * y
y_new = -sin_val * x + cos_val * y
```

## 重要发现

1. **CuPy只包装了最常用的21个cuBLAS函数**
2. **大量变体（Ex版本、batched版本）没有被包装**
3. **某些操作（如copy, swap, rot）虽然常用，但CuPy未直接包装**
4. **建议策略**：
   - 先生成有CuPy包装的函数（高质量baseline）
   - 再处理无CuPy包装的函数（使用CuPy数组操作）
"""

# 保存Markdown文件
output_path = '/share/project/zpy/flagbench/script/cublas_cupy/cupy_cublas_coverage.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n✓ Markdown报告已更新: {output_path}")
