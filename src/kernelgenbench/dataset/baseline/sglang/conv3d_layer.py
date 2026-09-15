"""
SGLang Conv3dLayer baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.conv import Conv3dLayer as _Conv3dLayer
except ModuleNotFoundError:
    _Conv3dLayer = None

_conv3d_module = None
_current_conv3d_config = None

def _get_conv3d_module(in_channels, out_channels, kernel_size, stride, padding, bias):
    global _conv3d_module, _current_conv3d_config
    key = (in_channels, out_channels, kernel_size, stride, padding, bias)
    if _current_conv3d_config != key:
        _conv3d_module = _Conv3dLayer(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        ).cuda()
        _current_conv3d_config = key
    return _conv3d_module

def conv3d_layer(x: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> torch.Tensor:
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
    return _conv3d_module(x)

