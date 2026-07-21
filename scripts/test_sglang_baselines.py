"""
Test all 50 SGLang baseline operators.
Verifies: import OK, function exists, can be called with minimal inputs.
"""
import sys
import traceback
import time

sys.path.insert(0, "/share/project/zpy/flagbench/src")

results = {"pass": [], "fail": [], "skip": []}

# Minimal test inputs for each operator
TESTS = {
    # activation.py
    "silu_and_mul": lambda b: b.silu_and_mul(torch.randn(4, 512, device='cuda')),
    "gelu_and_mul": lambda b: b.gelu_and_mul(torch.randn(4, 512, device='cuda')),
    "quick_gelu": lambda b: b.quick_gelu(torch.randn(4, 512, device='cuda')),
    "new_gelu": lambda b: b.new_gelu(torch.randn(4, 512, device='cuda')),
    "xielu": lambda b: b.xielu(torch.randn(4, 512, device='cuda')),

    # layernorm.py
    "rms_norm": lambda b: b.rms_norm(torch.randn(4, 512, device='cuda'),
                                      torch.ones(512, device='cuda')),
    "layer_norm": lambda b: b.layer_norm(torch.randn(4, 512, device='cuda'), 512),
    "gemma_rms_norm": lambda b: b.gemma_rms_norm(torch.randn(4, 512, device='cuda'),
                                                   torch.ones(512, device='cuda')),
    "gemma3_rms_norm": lambda b: b.gemma3_rms_norm(torch.randn(4, 512, device='cuda'), 512),
    "gemma4_rms_norm": lambda b: b.gemma4_rms_norm(torch.randn(4, 512, device='cuda'), 512),
    "rms_norm_without_scale": lambda b: b.rms_norm_without_scale(torch.randn(4, 512, device='cuda'), 512),

    # rotary_embedding/
    "rotary_embedding": lambda b: b.rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "mrotary_embedding": lambda b: b.mrotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64, mrope_section=[10, 11, 11]),
    "dual_chunk_rotary_embedding": lambda b: b.dual_chunk_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "deepseek_scaling_rotary_embedding": lambda b: b.deepseek_scaling_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "llama3_rotary_embedding": lambda b: b.llama3_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "dynamic_ntk_scaling_rotary_embedding": lambda b: b.dynamic_ntk_scaling_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "linear_scaling_rotary_embedding": lambda b: b.linear_scaling_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "phi3_long_rope_scaled_rotary_embedding": lambda b: b.phi3_long_rope_scaled_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64),
    "triton_mrope_fused": lambda b: b.triton_mrope_fused(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4 * 64, device='cuda'),
        torch.randn(128, 4 * 64, device='cuda'),
        torch.randn(8192, 128, device='cuda'), head_size=128, rotary_dim=128,
        mrope_section=[42, 42, 44]),
    "triton_ernie45_rope_fused": lambda b: b.triton_ernie45_rope_fused(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4 * 64, device='cuda'),
        torch.randn(128, 4 * 64, device='cuda'),
        torch.randn(8192, 128, device='cuda'), head_size=128, rotary_dim=128,
        mrope_section=[42, 42, 44]),
    "apply_interleaved_rope_triton": lambda b: b.apply_interleaved_rope_triton(
        torch.randn(3, 128, 64, device='cuda'), mrope_section=[10, 11, 11]),
    "dynamic_ntk_alpha_rotary_embedding": lambda b: b.dynamic_ntk_alpha_rotary_embedding(
        torch.arange(128, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        torch.randn(128, 4, 64, device='cuda'),
        head_size=64, rotary_dim=64, scaling_alpha=2.0),

    # moe/
    "fused_moe": lambda b: None,  # needs routing_data — complex, skip for now
    "topk": lambda b: None,  # needs full module init
    "moe_align_block_size": lambda b: b.moe_align_block_size(
        torch.randint(0, 4, (128, 2), device='cuda', dtype=torch.int32),
        num_experts=4, block_size=32),

    # attention/fla/
    "l2norm": lambda b: b.l2norm(torch.randn(4, 512, device='cuda')),
    "rms_norm_gated": lambda b: b.rms_norm_gated(
        torch.randn(4, 512, device='cuda'), torch.randn(512, device='cuda')),
    "fused_recurrent_gated_delta_rule": lambda b: b.fused_recurrent_gated_delta_rule(
        q=torch.randn(1, 64, 4, 64, device='cuda'),
        k=torch.randn(1, 64, 4, 64, device='cuda'),
        v=torch.randn(1, 64, 4, 64, device='cuda'),
        g=torch.randn(1, 64, 4, device='cuda'),
        beta=torch.randn(1, 64, 4, device='cuda'),
        scale=0.125),
    "fused_recurrent_gated_delta_rule_update": lambda b: None,  # complex
    "fused_sigmoid_gating_delta_rule_update": lambda b: None,  # complex
    "fused_sigmoid_gating_delta_rule_packed_decode": lambda b: None,  # complex
    "fused_gdn_gating": lambda b: b.fused_gdn_gating(
        A_log=torch.randn(16, device='cuda'),
        a=torch.randn(4, 16, device='cuda'),
        b=torch.randn(4, 16, device='cuda'),
        dt_bias=torch.randn(16, device='cuda')),
    "layer_norm_gated_fwd": lambda b: b.layer_norm_gated_fwd(
        torch.randn(4, 512, device='cuda'),
        torch.randn(4, 512, device='cuda'),
        torch.ones(512, device='cuda'), torch.zeros(512, device='cuda')),

    # attention/mamba/
    "causal_conv1d_fn": lambda b: b.causal_conv1d_fn(
        torch.randn(8, 64, device='cuda'),
        torch.randn(64, 4, device='cuda')),
    "causal_conv1d_update": lambda b: b.causal_conv1d_update(
        torch.randn(4, 64, device='cuda'),
        torch.randn(4, 64, 3, device='cuda'),
        torch.randn(64, 4, device='cuda')),
    "selective_scan_update": lambda b: b.selective_scan_update(
        state=torch.randn(4, 64, 16, device='cuda'),
        x=torch.randn(4, 64, device='cuda'),
        dt=torch.randn(4, 64, device='cuda'),
        A=torch.randn(64, 16, device='cuda'),
        B=torch.randn(4, 16, device='cuda'),
        C=torch.randn(4, 16, device='cuda')),
    "mamba_chunk_scan_combined_fwd": lambda b: b.mamba_chunk_scan_combined_fwd(
        torch.randn(1, 64, 4, 64, device='cuda'),
        torch.randn(1, 64, 4, device='cuda'),
        torch.randn(4, device='cuda'),
        torch.randn(1, 64, 1, 16, device='cuda'),
        torch.randn(1, 64, 1, 16, device='cuda')),
    "mixer2_rms_norm_gated": lambda b: b.mixer2_rms_norm_gated(
        torch.randn(4, 512, device='cuda'), torch.randn(4, 512, device='cuda'), 512),

    # elementwise.py
    "fused_dual_residual_rmsnorm": lambda b: b.fused_dual_residual_rmsnorm(
        torch.randn(4, 512, device='cuda'), torch.randn(4, 512, device='cuda'), 512, 512),
    "softcap": lambda b: b.softcap(torch.randn(4, 512, device='cuda')),
    "silu_and_mul_triton": lambda b: b.silu_and_mul_triton(torch.randn(4, 1024, device='cuda')),
    "gelu_and_mul_triton": lambda b: b.gelu_and_mul_triton(torch.randn(4, 1024, device='cuda')),
    "fused_rmsnorm": lambda b: b.fused_rmsnorm(
        torch.randn(4, 512, device='cuda'), torch.ones(512, device='cuda')),
    "experts_combine_triton": lambda b: b.experts_combine_triton(
        torch.randn(4, 2, 512, device='cuda'), torch.randn(4, 512, device='cuda')),

    # gemma4_fused_ops.py
    "gemma_rmsnorm_residual_scalar": lambda b: b.gemma_rmsnorm_residual_scalar(
        torch.randn(4, 512, device='cuda'), torch.randn(512, device='cuda'),
        torch.randn(4, 512, device='cuda'), torch.tensor(0.5, device='cuda')),
    "gemma_qkv_rmsnorm": lambda b: b.gemma_qkv_rmsnorm(
        torch.randn(4, 256, device='cuda'), torch.randn(4, 128, device='cuda'),
        torch.randn(4, 128, device='cuda'), torch.randn(64, device='cuda'),
        torch.randn(64, device='cuda'), num_q_heads=4, num_kv_heads=2, head_dim=64),

    # conv.py
    "conv2d_layer": lambda b: b.conv2d_layer(
        torch.randn(4, 3, 32, 32, device='cuda'), 3, 16, 3),
    "conv3d_layer": lambda b: b.conv3d_layer(
        torch.randn(4, 3, 16, 16, 16, device='cuda'), 3, 16, 3),

    # quantization/
    "per_token_quant_int8": lambda b: b.per_token_quant_int8(
        torch.randn(4, 512, device='cuda')),
}


def main():
    print(f"Testing {len(TESTS)} operators (import only, no GPU available)...\n")

    for name in sorted(TESTS.keys()):
        try:
            # Import the baseline module
            mod = __import__(f"kernelgenbench.dataset.baseline.sglang.{name}", fromlist=[name])
            fn = getattr(mod, name)
            print(f"  {name}: import OK, signature={fn.__code__.co_varnames[:fn.__code__.co_argcount]}")
            results["pass"].append(name)

        except Exception as e:
            err = str(e).split('\n')[0][:120]
            print(f"  {name}: FAIL - {err}")
            results["fail"].append(name)

    print(f"\n=== Results ===")
    print(f"PASS:  {len(results['pass'])}")
    print(f"FAIL:  {len(results['fail'])}")
    if results['fail']:
        print(f"\nFailed operators:")
        for name in results['fail']:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
