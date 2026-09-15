"""
SGLang apply_interleaved_rope_triton baseline.
Source: sglang.srt.layers.rotary_embedding.mrope.apply_interleaved_rope_triton(x, mrope_section)
"""
import torch
import typing

try:
    from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope_triton as _apply_interleaved_rope
except ModuleNotFoundError:
    _apply_interleaved_rope = None


def apply_interleaved_rope_triton(
    x: torch.Tensor,
    mrope_section: typing.List[int],
) -> torch.Tensor:
    """
    Rearranges interleaved RoPE values along dim=0 using mrope_section.
    x shape: [3, N, D] -> returns [N, D]
    """
    return _apply_interleaved_rope(x, mrope_section)
