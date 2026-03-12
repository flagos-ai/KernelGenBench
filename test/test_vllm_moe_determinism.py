"""
测试 vllm moe_align_block_size C++ 实现的确定性

结论：sorted_token_ids 在多次调用间不确定，experts_ids 和 num_tokens_post_pad 确定。
"""
import os
os.environ['DISPATCH_TORCH_LIB'] = '0'

import torch
from vllm import _custom_ops

device = 'cuda'

def run_baseline(topk_ids, num_experts, block_size, topk, num_tokens):
    max_out = num_tokens * topk + num_experts * block_size
    s = torch.empty(max_out, device=device, dtype=torch.int32)
    s.fill_(num_tokens * topk)
    e = torch.zeros(max_out, device=device, dtype=torch.int32)
    n = torch.empty(1, device=device, dtype=torch.int32)
    _custom_ops.moe_align_block_size(topk_ids.clone(), num_experts, block_size, s, e, n, None)
    return s, e, n

configs = [
    (128, 4, 16, 2),
    (1024, 8, 16, 2),
    (4096, 16, 64, 4),
]

print("=" * 70)
print("vllm moe_align_block_size 确定性测试")
print("同一个 baseline 函数，同一个输入，调用两次，比较结果")
print("=" * 70)

for num_tokens, num_experts, block_size, topk in configs:
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

    s1, e1, n1 = run_baseline(topk_ids, num_experts, block_size, topk, num_tokens)
    s2, e2, n2 = run_baseline(topk_ids, num_experts, block_size, topk, num_tokens)

    sorted_match = torch.equal(s1, s2)
    expert_match = torch.equal(e1, e2)
    npad_match = torch.equal(n1, n2)

    print(f"\nconfig: num_tokens={num_tokens}, num_experts={num_experts}, "
          f"block_size={block_size}, topk={topk}")
    print(f"  sorted_token_ids  一致: {sorted_match}")
    print(f"  experts_ids       一致: {expert_match}")
    print(f"  num_tokens_post_pad 一致: {npad_match}")

    if not sorted_match:
        diff_count = (s1 != s2).sum().item()
        print(f"  sorted_token_ids 不一致元素数: {diff_count}/{s1.numel()}")
