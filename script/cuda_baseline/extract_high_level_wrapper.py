"""
提取Kaldi的高层wrapper函数（包含grid/block计算逻辑）

策略：直接从cu-matrix.cc等文件提取完整的wrapper函数，
包含grid/block配置计算，这样Python接口更简洁。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KaldiHighLevelExtractor:
    """
    从Kaldi的C++源码中提取高层wrapper函数
    
    目标：提取包含完整逻辑的wrapper，例如：
    
    void CuMatrixBase<Real>::CopyLowerToUpper() {
        dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
        dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK), ...);
        cuda_copy_low_upp(dimGrid, dimBlock, data_, Dim());
    }
    
    转换为可以从torch.Tensor调用的版本。
    """
    
    def __init__(self, kaldi_repo_path: str = "/share/project/zpy/k1_repo"):
        self.kaldi_repo = Path(kaldi_repo_path)
        self.src_dir = self.kaldi_repo / "src" / "cudamatrix"
        
        if not self.src_dir.exists():
            raise ValueError(f"Kaldi source directory not found: {self.src_dir}")
        
        logger.info(f"KaldiHighLevelExtractor initialized with repo: {self.kaldi_repo}")
    
    def find_wrapper_function(self, kernel_name: str) -> Optional[Dict]:
        """
        查找kernel对应的高层wrapper函数
        
        例如：copy_low_upp -> CuMatrixBase::CopyLowerToUpper()
        
        Returns:
            {
                'function_name': 'CopyLowerToUpper',
                'class_name': 'CuMatrixBase',
                'source_file': 'cu-matrix.cc',
                'code': '完整函数代码',
                'line_start': 起始行号,
                'line_end': 结束行号
            }
        """
        # 策略：搜索调用cuda_xxx或cudaF_xxx的函数
        # 注意：cu-matrix.cc中调用的是cuda_xxx（通用wrapper），不是cudaF/cudaD
        pattern = f"cuda_{kernel_name}"
        
        # 在.cc文件中搜索
        for cc_file in self.src_dir.glob("*.cc"):
            content = cc_file.read_text()
            
            # 查找调用pattern的函数
            if pattern in content:
                logger.info(f"Found {pattern} in {cc_file.name}")
                
                # 提取包含该调用的函数
                func_info = self._extract_function_containing_call(
                    content, pattern, cc_file.name
                )
                
                if func_info:
                    return func_info
        
        return None
    
    def _extract_function_containing_call(
        self, content: str, call_pattern: str, filename: str
    ) -> Optional[Dict]:
        """
        从源码中提取包含特定调用的函数
        """
        lines = content.split('\n')
        
        # 找到调用所在行
        call_line_idx = None
        for i, line in enumerate(lines):
            if call_pattern in line:
                call_line_idx = i
                break
        
        if call_line_idx is None:
            return None
        
        # 向上查找函数签名
        func_start_idx = None
        brace_count = 0
        
        for i in range(call_line_idx, -1, -1):
            line = lines[i].strip()
            
            # 查找函数开始（包含函数名和{）
            if '{' in line:
                # 继续向上查找函数签名的开始
                for j in range(i, -1, -1):
                    test_line = lines[j].strip()
                    # 函数签名通常以类型或template开始
                    if (test_line.startswith('void ') or 
                        test_line.startswith('template') or
                        test_line.startswith('Real ') or
                        test_line.startswith('int ') or
                        test_line.startswith('bool ')):
                        func_start_idx = j
                        break
                break
        
        if func_start_idx is None:
            return None
        
        # 向下查找函数结束（配对的}）
        func_end_idx = None
        brace_count = 0
        in_function = False
        
        for i in range(func_start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1
                    if in_function and brace_count == 0:
                        func_end_idx = i
                        break
            if func_end_idx is not None:
                break
        
        if func_end_idx is None:
            return None
        
        # 提取函数代码
        func_code = '\n'.join(lines[func_start_idx:func_end_idx+1])
        
        # 解析函数名
        func_name = self._parse_function_name(func_code)
        class_name = self._parse_class_name(func_code)
        
        return {
            'function_name': func_name,
            'class_name': class_name,
            'source_file': filename,
            'code': func_code,
            'line_start': func_start_idx + 1,
            'line_end': func_end_idx + 1,
            'call_pattern': call_pattern
        }
    
    def _parse_function_name(self, func_code: str) -> str:
        """从函数代码中解析函数名"""
        # 匹配 ClassName::FunctionName 或 FunctionName
        match = re.search(r'(\w+::)?(\w+)\s*\(', func_code)
        if match:
            return match.group(2)
        return "Unknown"
    
    def _parse_class_name(self, func_code: str) -> Optional[str]:
        """从函数代码中解析类名"""
        match = re.search(r'(\w+)::', func_code)
        if match:
            return match.group(1)
        return None
    
    def extract_kernel_list(self) -> List[str]:
        """
        从cu-kernels-ansi.h提取所有kernel名称
        """
        header_file = self.src_dir / "cu-kernels-ansi.h"
        content = header_file.read_text()
        
        # 匹配 void cudaF_xxx 或 void cudaD_xxx
        pattern = r'void cuda[FD]_(\w+)\s*\('
        matches = re.findall(pattern, content)
        
        # 去重
        kernels = sorted(set(matches))
        logger.info(f"Found {len(kernels)} unique kernels in cu-kernels-ansi.h")
        
        return kernels


def test_extractor():
    """测试提取器"""
    print("="*60)
    print("Testing KaldiHighLevelExtractor")
    print("="*60)
    
    extractor = KaldiHighLevelExtractor()
    
    # 测试：查找copy_low_upp的wrapper
    print("\n" + "="*60)
    print("Test 1: Find wrapper for 'copy_low_upp'")
    print("="*60)
    
    result = extractor.find_wrapper_function('copy_low_upp')
    
    if result:
        print(f"✓ Found wrapper function:")
        print(f"  Function: {result['class_name']}::{result['function_name']}")
        print(f"  File: {result['source_file']}:{result['line_start']}-{result['line_end']}")
        print(f"  Call pattern: {result['call_pattern']}")
        print(f"\nCode preview (first 500 chars):")
        print("-" * 60)
        print(result['code'][:500])
        print("-" * 60)
    else:
        print("✗ Wrapper not found")
    
    # 测试：列出所有kernels
    print("\n" + "="*60)
    print("Test 2: List all kernels")
    print("="*60)
    
    kernels = extractor.extract_kernel_list()
    print(f"Total kernels: {len(kernels)}")
    print(f"First 20: {kernels[:20]}")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extractor()
