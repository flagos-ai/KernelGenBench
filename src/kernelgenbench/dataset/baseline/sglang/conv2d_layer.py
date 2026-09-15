"""
SGLang Conv2dLayer baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.conv import Conv2dLayer as _Conv2dLayer
except ModuleNotFoundError:
    _Conv2dLayer = None

_conv2d_module = None
_current_conv2d_config = None

def _get_conv2d_module(in_channels, out_channels, kernel_size, stride, padding, bias):
    global _conv2d_module, _current_conv2d_config
    key = (in_channels, out_channels, kernel_size, stride, padding, bias)
    if _current_conv2d_config != key:
        _conv2d_module = _Conv2dLayer(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        ).cuda()
        _current_conv2d_config = key
    return _conv2d_module

def conv2d_layer(x: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        padding: int
        bias: bool
    Returns:
        torch.Tensor
    """
    return _conv2d_module(x)

