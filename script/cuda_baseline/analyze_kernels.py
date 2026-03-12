"""直接解析 kernel_list_k1.py 找到简单的kernels"""
import ast

# 读取文件
with open('/share/project/zpy/flagbench/src/flagbench/dataset/kernel_list_k1.py', 'r') as f:
    content = f.read()

# 解析AST
tree = ast.parse(content)

# 找到 IMPL_INFO_K1 字典
impl_info_k1 = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'IMPL_INFO_K1':
                impl_info_k1 = ast.literal_eval(node.value)
                break

if impl_info_k1 is None:
    print("ERROR: Could not find IMPL_INFO_K1")
    exit(1)

print(f"Total kernels in IMPL_INFO_K1: {len(impl_info_k1)}")

# 找出简单的 torch.Tensor 接口 kernels
simple_kernels = []
for name, info in impl_info_k1.items():
    input_args = info.get('input_args', [])
    
    # 检查是否所有输入都是 torch.Tensor 或 scalar
    simple_types = {'torch.Tensor', 'float', 'int', 'double', 'bool'}
    all_simple = all(arg.get('type') in simple_types for arg in input_args)
    
    if all_simple and len(input_args) >= 1:
        # 计算tensor和scalar数量
        tensor_count = sum(1 for arg in input_args if arg.get('type') == 'torch.Tensor')
        scalar_count = len(input_args) - tensor_count
        
        simple_kernels.append({
            'name': name,
            'num_args': len(input_args),
            'tensor_count': tensor_count,
            'scalar_count': scalar_count,
            'args': input_args,
            'desc': info.get('description', '')
        })

# 按复杂度排序（先tensor数量，再scalar数量）
simple_kernels.sort(key=lambda x: (x['tensor_count'], x['scalar_count']))

print(f"\nFound {len(simple_kernels)} kernels with torch.Tensor/scalar interface")
print("\nFirst 40 (sorted by complexity):")
print("="*80)

for i, k in enumerate(simple_kernels[:40]):
    arg_str = ', '.join([f"{a['name']}:{a['type']}" for a in k['args']])
    print(f"{i+1:3d}. {k['name']:30s} | T={k['tensor_count']} S={k['scalar_count']} | {k['desc'][:50]}")
