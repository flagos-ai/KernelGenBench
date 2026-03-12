#!/usr/bin/env python3
"""
测试 rmsnorm_backward 是否正确实现
基于用户提供的测试用例
"""

import sys
import os
import torch

# 尝试导入 flag_gems 和 flagbench
try:
    import flag_gems
    HAS_FLAG_GEMS = True
except ImportError:
    HAS_FLAG_GEMS = False
    print("⚠️  flag_gems 未安装，将只测试 flagbench")

try:
    import flagbench
    from sandbox.register import REGISTERED_OPS
    HAS_FLAGBENCH = True
except ImportError:
    HAS_FLAGBENCH = False
    print("⚠️  flagbench 未正确导入")


def test_native():
    """测试原生 PyTorch 实现"""
    print("\n" + "="*80)
    print("测试 1: 原生 PyTorch rms_norm")
    print("="*80)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    inp_file = ".input_cache.pt"
    if not os.path.exists(inp_file):
        print(f"❌ 错误: 找不到输入文件 {inp_file}")
        print("   请确保 .input_cache.pt 文件在当前目录")
        return None
    
    data = torch.load(inp_file)
    x = data["x"].cuda()
    x.retain_grad()
    w = data["w"].cuda()
    w.retain_grad()
    
    y = torch.nn.functional.rms_norm(x, (512,), w, eps=1e-5)
    y.sum().backward()
    
    print(f"✅ native norm, x.grad shape: {x.grad.shape}")
    print(f"   x.grad stats: min={x.grad.min().item():.6f}, max={x.grad.max().item():.6f}, mean={x.grad.mean().item():.6f}")
    print(f"   w.grad shape: {w.grad.shape}")
    print(f"   w.grad stats: min={w.grad.min().item():.6f}, max={w.grad.max().item():.6f}, mean={w.grad.mean().item():.6f}")
    
    return x.grad.clone(), w.grad.clone()


def test_flag_gems():
    """测试 flag_gems 实现"""
    if not HAS_FLAG_GEMS:
        print("\n⚠️  跳过 flag_gems 测试（未安装）")
        return None
    
    print("\n" + "="*80)
    print("测试 2: flag_gems rms_norm")
    print("="*80)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    flag_gems.enable()
    
    inp_file = ".input_cache.pt"
    data = torch.load(inp_file)
    x = data["x"].cuda()
    x.retain_grad()
    w = data["w"].cuda()
    w.retain_grad()
    
    y = torch.nn.functional.rms_norm(x, (512,), w, eps=1e-5)
    y.sum().backward()
    
    print(f"✅ gems norm, x.grad shape: {x.grad.shape}")
    print(f"   x.grad stats: min={x.grad.min().item():.6f}, max={x.grad.max().item():.6f}, mean={x.grad.mean().item():.6f}")
    print(f"   w.grad shape: {w.grad.shape}")
    print(f"   w.grad stats: min={w.grad.min().item():.6f}, max={w.grad.max().item():.6f}, mean={w.grad.mean().item():.6f}")
    
    return x.grad.clone(), w.grad.clone()


def test_flagbench():
    """测试 flagbench 实现"""
    if not HAS_FLAGBENCH:
        print("\n⚠️  跳过 flagbench 测试（未正确导入）")
        return None
    
    print("\n" + "="*80)
    print("测试 3: flagbench rms_norm")
    print("="*80)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    inp_file = ".input_cache.pt"
    data = torch.load(inp_file)
    x = data["x"].cuda()
    x.retain_grad()
    w = data["w"].cuda()
    w.retain_grad()
    
    with flagbench.use_gems(REGISTERED_OPS):
        y = torch.nn.functional.rms_norm(x, (512,), w, eps=1e-5)
    y.sum().backward()
    
    print(f"✅ flagbench norm, x.grad shape: {x.grad.shape}")
    print(f"   x.grad stats: min={x.grad.min().item():.6f}, max={x.grad.max().item():.6f}, mean={x.grad.mean().item():.6f}")
    print(f"   w.grad shape: {w.grad.shape}")
    print(f"   w.grad stats: min={w.grad.min().item():.6f}, max={w.grad.max().item():.6f}, mean={w.grad.mean().item():.6f}")
    
    return x.grad.clone(), w.grad.clone()


def compare_grads(grad1, grad2, name1="Grad 1", name2="Grad 2", rtol=1e-5, atol=1e-6):
    """比较两个梯度"""
    if grad1 is None or grad2 is None:
        return False
    
    print(f"\n📊 比较 {name1} vs {name2}:")
    
    # 检查形状
    if grad1.shape != grad2.shape:
        print(f"❌ 形状不匹配: {grad1.shape} vs {grad2.shape}")
        return False
    
    # 计算差异
    diff = grad1 - grad2
    max_diff = diff.abs().max().item()
    mean_diff = diff.abs().mean().item()
    
    # 使用 torch.allclose 检查
    is_close = torch.allclose(grad1, grad2, rtol=rtol, atol=atol)
    
    print(f"   最大差异: {max_diff:.2e}")
    print(f"   平均差异: {mean_diff:.2e}")
    print(f"   是否接近 (rtol={rtol}, atol={atol}): {'✅ 是' if is_close else '❌ 否'}")
    
    if not is_close:
        # 找出差异最大的位置
        max_idx = diff.abs().argmax()
        max_idx_tuple = tuple(torch.unravel_index(max_idx, grad1.shape))
        print(f"   最大差异位置: {max_idx_tuple}")
        print(f"   {name1}[{max_idx_tuple}] = {grad1[max_idx_tuple].item():.6f}")
        print(f"   {name2}[{max_idx_tuple}] = {grad2[max_idx_tuple].item():.6f}")
    
    return is_close


def main():
    print("="*80)
    print("RMSNorm Backward 测试脚本")
    print("="*80)
    
    # 检查输入文件
    inp_file = ".input_cache.pt"
    if not os.path.exists(inp_file):
        print(f"\n❌ 错误: 找不到输入文件 {inp_file}")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"   请确保 .input_cache.pt 文件在当前目录")
        return 1
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("\n❌ 错误: CUDA 不可用")
        return 1
    
    print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 运行测试
    native_x_grad, native_w_grad = test_native()
    gems_x_grad, gems_w_grad = test_flag_gems()
    flagbench_x_grad, flagbench_w_grad = test_flagbench()
    
    # 比较结果
    print("\n" + "="*80)
    print("结果比较")
    print("="*80)
    
    all_passed = True
    
    if gems_x_grad is not None and native_x_grad is not None:
        x_match = compare_grads(native_x_grad, gems_x_grad, "Native x.grad", "flag_gems x.grad")
        w_match = compare_grads(native_w_grad, gems_w_grad, "Native w.grad", "flag_gems w.grad")
        if not (x_match and w_match):
            all_passed = False
    
    if flagbench_x_grad is not None and native_x_grad is not None:
        x_match = compare_grads(native_x_grad, flagbench_x_grad, "Native x.grad", "flagbench x.grad")
        w_match = compare_grads(native_w_grad, flagbench_w_grad, "Native w.grad", "flagbench w.grad")
        if not (x_match and w_match):
            all_passed = False
    
    if gems_x_grad is not None and flagbench_x_grad is not None:
        x_match = compare_grads(gems_x_grad, flagbench_x_grad, "flag_gems x.grad", "flagbench x.grad")
        w_match = compare_grads(gems_w_grad, flagbench_w_grad, "flag_gems w.grad", "flagbench w.grad")
        if not (x_match and w_match):
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败，请检查上述差异")
        return 1


if __name__ == "__main__":
    sys.exit(main())

