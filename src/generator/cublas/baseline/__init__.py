"""
Baseline generation module for cuBLAS functions.

This module contains tools for generating baseline implementations
of cuBLAS functions using CuPy.
"""

from .cublas_c_api_config import CUBLAS_C_API_CONFIG

__all__ = ['CUBLAS_C_API_CONFIG']
