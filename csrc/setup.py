# Auto-generated setup.py for Kaldi CUDA bindings
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Kaldi source directory
KALDI_SRC = os.path.join(os.path.dirname(__file__), "../kaldi/src")

setup(
    name='kaldi_ops',
    ext_modules=[
        CUDAExtension(
            name='kaldi_ops',
            sources=['kaldi_ops.cpp'],
            include_dirs=[
                KALDI_SRC,
                os.path.join(KALDI_SRC, "cudamatrix"),
            ],
            library_dirs=[
                os.path.join(KALDI_SRC, "cudamatrix"),
            ],
            libraries=['kaldi-cudamatrix'],  # Link against libkaldi-cudamatrix
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
