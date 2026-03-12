#!/usr/bin/env python3
"""
测试 rmsnorm_backward 是否正确实现
直接使用用户提供的测试用例
"""

import sys
import os
import torch
import flag_gems


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # native
    inp_file = ".input_cache.pt"
    if not os.path.exists(inp_file):
        print(f"❌ 错误: 找不到输入文件 {inp_file}")
        print(f"   当前工作目录: {os.getcwd()}")
        return 1
    
    data = torch.load(inp_file)
    x = data["x"].cuda()
    x.retain_grad()
    w = data["w"].cuda()
    w.retain_grad()
    y = torch.nn.functional.rms_norm(x, (512,), w, eps=1e-5)
    y.sum().backward()
    native_x_grad = x.grad.clone()
    print(f"native norm, x.grad shape={x.grad.shape}, min={x.grad.min().item():.6f}, max={x.grad.max().item():.6f}, mean={x.grad.mean().item():.6f}")
    
    # gems
    flag_gems.enable()
    inp_file = ".input_cache.pt"
    data = torch.load(inp_file)
    x = data["x"].cuda()
    x.retain_grad()
    w = data["w"].cuda()
    w.retain_grad()
    y = torch.nn.functional.rms_norm(x, (512,), w, eps=1e-5)
    y.sum().backward()
    gems_x_grad = x.grad.clone()
    print(f"gems norm, x.grad shape={x.grad.shape}, min={x.grad.min().item():.6f}, max={x.grad.max().item():.6f}, mean={x.grad.mean().item():.6f}")
    
    # 比较结果
    if torch.allclose(native_x_grad, gems_x_grad, rtol=1e-5, atol=1e-6):
        print("\n✅ 测试通过！flag_gems 的梯度与原生 PyTorch 一致")
        return 0
    else:
        max_diff = (native_x_grad - gems_x_grad).abs().max().item()
        mean_diff = (native_x_grad - gems_x_grad).abs().mean().item()
        print(f"\n❌ 测试失败！最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

