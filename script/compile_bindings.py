#!/usr/bin/env python3
"""
阶段3: 编译PyTorch C++ extension

功能：
1. 检查Kaldi库是否已编译
2. 如果需要，编译Kaldi CUDA库
3. 编译PyTorch C++ extension
4. 验证编译成功

使用方法：
python script/compile_bindings.py \
    --csrc-dir csrc \
    --kaldi-src kaldi/src \
    --output lib/kaldi_ops.so
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


class BindingCompiler:
    """编译PyTorch bindings"""
    
    def __init__(self, csrc_dir: Path, kaldi_src: Path, output_dir: Path):
        self.csrc_dir = csrc_dir
        self.kaldi_src = kaldi_src
        self.output_dir = output_dir
        
    def check_kaldi_library(self) -> bool:
        """检查Kaldi库是否存在"""
        # 查找libkaldi-cudamatrix.so或.a
        cudamatrix_dir = self.kaldi_src / "cudamatrix"
        
        lib_files = list(cudamatrix_dir.glob("libkaldi-cudamatrix.so*"))
        lib_files.extend(cudamatrix_dir.glob("libkaldi-cudamatrix.a"))
        
        if lib_files:
            print(f"✓ Found Kaldi library: {lib_files[0]}")
            return True
        else:
            print("✗ Kaldi library not found")
            return False
    
    def compile_kaldi(self) -> bool:
        """编译Kaldi CUDA库（如果需要）"""
        print("\n=== Compiling Kaldi CUDA library ===")
        
        cudamatrix_dir = self.kaldi_src / "cudamatrix"
        
        if not cudamatrix_dir.exists():
            print(f"Error: Kaldi source directory not found: {cudamatrix_dir}")
            return False
        
        # 检查是否有Makefile
        makefile = cudamatrix_dir / "Makefile"
        if not makefile.exists():
            print(f"Warning: Makefile not found in {cudamatrix_dir}")
            print("Assuming Kaldi needs to be configured first...")
            print("Please run the following commands manually:")
            print(f"  cd {self.kaldi_src.parent}")
            print(f"  ./configure")
            print(f"  cd src")
            print(f"  make cudamatrix")
            return False
        
        # 尝试编译
        try:
            print(f"Running: make -C {cudamatrix_dir}")
            result = subprocess.run(
                ["make"],
                cwd=cudamatrix_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                print("✓ Kaldi library compiled successfully")
                return True
            else:
                print(f"✗ Kaldi compilation failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Kaldi compilation timed out")
            return False
        except Exception as e:
            print(f"✗ Error compiling Kaldi: {e}")
            return False
    
    def compile_pytorch_extension(self) -> bool:
        """编译PyTorch C++ extension"""
        print("\n=== Compiling PyTorch extension ===")
        
        setup_py = self.csrc_dir / "setup.py"
        
        if not setup_py.exists():
            print(f"Error: setup.py not found: {setup_py}")
            return False
        
        # 运行setup.py build
        try:
            print(f"Running: python {setup_py} build_ext --inplace")
            
            env = os.environ.copy()
            # 确保CUDA可见
            if "CUDA_HOME" not in env and "CUDA_PATH" not in env:
                cuda_paths = ["/usr/local/cuda", "/opt/cuda"]
                for cuda_path in cuda_paths:
                    if Path(cuda_path).exists():
                        env["CUDA_HOME"] = cuda_path
                        print(f"Setting CUDA_HOME={cuda_path}")
                        break
            
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=self.csrc_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=900  # 15分钟超时
            )
            
            if result.returncode == 0:
                print("✓ PyTorch extension compiled successfully")
                print(result.stdout)
                return True
            else:
                print("✗ PyTorch extension compilation failed:")
                print(result.stderr)
                print(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Compilation timed out")
            return False
        except Exception as e:
            print(f"✗ Error compiling extension: {e}")
            return False
    
    def find_compiled_library(self) -> Path:
        """查找编译好的.so文件"""
        # 查找build目录下的.so文件
        build_dirs = [
            self.csrc_dir / "build",
            self.csrc_dir,
        ]
        
        for build_dir in build_dirs:
            if build_dir.exists():
                so_files = list(build_dir.rglob("*.so"))
                if so_files:
                    return so_files[0]
        
        return None
    
    def copy_to_output(self, so_file: Path) -> bool:
        """复制.so文件到输出目录"""
        if not so_file or not so_file.exists():
            print(f"Error: Compiled library not found")
            return False
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / so_file.name
        
        try:
            import shutil
            shutil.copy2(so_file, output_file)
            print(f"✓ Copied library to: {output_file}")
            return True
        except Exception as e:
            print(f"✗ Error copying library: {e}")
            return False
    
    def compile(self) -> bool:
        """执行完整的编译流程"""
        print("="*60)
        print("Stage 3: Compiling Bindings")
        print("="*60)
        
        # 1. 检查Kaldi库
        if not self.check_kaldi_library():
            print("\nKaldi library not found, attempting to compile...")
            if not self.compile_kaldi():
                print("\n⚠ Warning: Kaldi compilation failed or needs manual setup")
                print("You may need to compile Kaldi manually first.")
                # 继续尝试，可能已经有编译好的库
        
        # 2. 编译PyTorch extension
        if not self.compile_pytorch_extension():
            print("\n✗ Stage 3 failed: PyTorch extension compilation failed")
            return False
        
        # 3. 查找并复制.so文件
        so_file = self.find_compiled_library()
        if so_file:
            if self.copy_to_output(so_file):
                print(f"\n✓ Stage 3 completed successfully!")
                print(f"   Library: {self.output_dir / so_file.name}")
                return True
        else:
            print("\n⚠ Warning: Could not find compiled .so file")
            print("   Extension may have been built in a non-standard location")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Compile PyTorch C++ bindings for Kaldi CUDA kernels"
    )
    parser.add_argument(
        "--csrc-dir",
        type=Path,
        default=Path("csrc"),
        help="Directory containing generated C++ code"
    )
    parser.add_argument(
        "--kaldi-src",
        type=Path,
        default=Path("kaldi/src"),
        help="Kaldi source directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lib"),
        help="Output directory for compiled library"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    csrc_dir = project_root / args.csrc_dir
    kaldi_src = project_root / args.kaldi_src
    output_dir = project_root / args.output_dir
    
    # 执行编译
    compiler = BindingCompiler(csrc_dir, kaldi_src, output_dir)
    success = compiler.compile()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
