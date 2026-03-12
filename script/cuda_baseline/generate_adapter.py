"""
Adapter Generator - 自动生成 PyTorch CUDA Adapter 代码

根据 kernel 信息自动生成 adapter_source，将 torch::Tensor 转换为
原始 CUDA 指针，并调用 launch_xxx 函数。

作者: FlagBench Team
日期: 2026-01-16
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TensorParam:
    """Tensor 参数信息"""
    name: str
    is_const: bool = False
    is_output: bool = False


@dataclass
class ScalarParam:
    """标量参数信息"""
    name: str
    type: str  # "float", "int", "double"


class AdapterGenerator:
    """
    自动生成 PyTorch C++ adapter 代码
    
    使用方式：
        gen = AdapterGenerator()
        adapter = gen.generate_adapter(
            kernel_name="copy_low_upp",
            tensor_params=[TensorParam("A", is_output=True)],
            scalar_params=[],
            grid_config="simple_2d"
        )
    """
    
    def __init__(self):
        self.grid_templates = {
            "simple_2d": self._generate_simple_2d_grid,
            "simple_1d": self._generate_simple_1d_grid,
            "custom": lambda **kw: kw.get("custom_grid_code", "")
        }
    
    def _generate_simple_2d_grid(self, **kwargs) -> str:
        """生成 2D grid 配置代码（用于矩阵操作）"""
        return """
    // 2D grid configuration (16x16 blocks)
    const int BLOCK_SIZE = 16;
    dim3 Bl(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Gr((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
"""
    
    def _generate_simple_1d_grid(self, **kwargs) -> str:
        """生成 1D grid 配置代码（用于向量操作）"""
        return """
    // 1D grid configuration
    const int BLOCK_SIZE = 256;
    dim3 Bl(BLOCK_SIZE);
    dim3 Gr((total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);
"""
    
    def generate_tensor_extraction(self, params: List[TensorParam]) -> str:
        """生成 tensor 参数提取代码"""
        lines = []
        for param in params:
            lines.append(f"    float* {param.name}_ptr = {param.name}.data_ptr<float>();")
        return "\n".join(lines)
    
    def generate_matrixdim_extraction(self, tensor_name: str = "A") -> str:
        """生成 MatrixDim 提取代码"""
        return f"""
    // Extract MatrixDim
    MatrixDim dim;
    dim.rows = {tensor_name}.size(0);
    dim.cols = {tensor_name}.size(1);
    dim.stride = {tensor_name}.stride(0);
"""
    
    def generate_checks(self, tensor_params: List[TensorParam]) -> str:
        """生成输入检查代码"""
        lines = ["    // Input validation"]
        for param in tensor_params:
            lines.append(f'    TORCH_CHECK({param.name}.device().is_cuda(), "{param.name} must be on CUDA");')
            lines.append(f'    TORCH_CHECK({param.name}.is_contiguous(), "{param.name} must be contiguous");')
        return "\n".join(lines)
    
    def generate_adapter(
        self,
        kernel_name: str,
        tensor_params: List[TensorParam],
        scalar_params: List[ScalarParam],
        grid_config: str = "simple_2d",
        custom_grid_code: str = "",
        need_matrixdim: bool = True,
        description: str = "",
    ) -> str:
        """
        生成完整的 adapter 代码
        
        Args:
            kernel_name: kernel 名称（如 "copy_low_upp"）
            tensor_params: Tensor 参数列表
            scalar_params: 标量参数列表
            grid_config: Grid 配置类型 ("simple_2d", "simple_1d", "custom")
            custom_grid_code: 自定义 grid 代码（当 grid_config="custom" 时使用）
            need_matrixdim: 是否需要 MatrixDim
            description: 函数描述
        
        Returns:
            完整的 adapter 源码字符串
        """
        # 生成函数签名
        param_list = []
        for tp in tensor_params:
            param_list.append(f"torch::Tensor {tp.name}")
        for sp in scalar_params:
            param_list.append(f"{sp.type} {sp.name}")
        
        func_signature = f"void {kernel_name}({', '.join(param_list)})"
        
        # 生成 launcher 参数签名
        launcher_params = ["dim3 Gr", "dim3 Bl"]
        for tp in tensor_params:
            const_prefix = "const " if tp.is_const else ""
            launcher_params.append(f"{const_prefix}float* {tp.name}")
        
        if need_matrixdim:
            # 假设第一个 tensor 用于 MatrixDim
            launcher_params.append("MatrixDim dim")
        
        for sp in scalar_params:
            launcher_params.append(f"{sp.type} {sp.name}")
        
        launcher_signature = f"extern \"C\" void launch_{kernel_name}({', '.join(launcher_params)});"
        
        # 生成函数体
        checks = self.generate_checks(tensor_params)
        
        # 提取维度信息（假设第一个是主 tensor）
        if tensor_params:
            first_tensor = tensor_params[0].name
            dim_code = f"""
    int rows = {first_tensor}.size(0);
    int cols = {first_tensor}.size(1);"""
        else:
            dim_code = ""
        
        # Grid 配置
        grid_gen = self.grid_templates.get(grid_config)
        if grid_gen:
            grid_code = grid_gen(custom_grid_code=custom_grid_code)
        else:
            grid_code = custom_grid_code
        
        # 提取指针
        tensor_extract = self.generate_tensor_extraction(tensor_params)
        
        # MatrixDim
        matrixdim_code = ""
        if need_matrixdim and tensor_params:
            matrixdim_code = self.generate_matrixdim_extraction(tensor_params[0].name)
        
        # 生成调用
        call_args = ["Gr", "Bl"]
        for tp in tensor_params:
            call_args.append(f"{tp.name}_ptr")
        if need_matrixdim:
            call_args.append("dim")
        for sp in scalar_params:
            call_args.append(sp.name)
        
        launcher_call = f"launch_{kernel_name}({', '.join(call_args)});"
        
        # 组装完整代码
        doc_comment = f"// {description}\n" if description else ""
        
        adapter_code = f'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// MatrixDim structure (must match CUDA side)
typedef struct MatrixDim_ {{
    int rows;
    int cols;
    int stride;
}} MatrixDim;

// Forward declaration of launcher
{launcher_signature}

{doc_comment}{func_signature} {{
{checks}
{dim_code}
{grid_code}

    // Extract tensor pointers
{tensor_extract}
{matrixdim_code}
    // Call CUDA launcher
    {launcher_call}
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{kernel_name}", &{kernel_name}, "{description}");
}}
'''
        return adapter_code


def generate_kaldi_k1_adapters():
    """
    为 Kaldi K1 的 3 个测试 kernel 生成 adapter
    """
    print("=" * 80)
    print("Adapter Generator - Kaldi K1 Kernels")
    print("=" * 80)
    
    gen = AdapterGenerator()
    
    # 1. copy_low_upp: void(torch::Tensor A)
    print("\n[1] Generating adapter for: copy_low_upp")
    copy_low_upp_adapter = gen.generate_adapter(
        kernel_name="copy_low_upp",
        tensor_params=[TensorParam("A", is_output=True)],
        scalar_params=[],
        grid_config="simple_2d",
        need_matrixdim=True,
        description="Copy lower triangle to upper triangle of square matrix"
    )
    
    # 2. copy_upp_low: void(torch::Tensor A)
    print("[2] Generating adapter for: copy_upp_low")
    copy_upp_low_adapter = gen.generate_adapter(
        kernel_name="copy_upp_low",
        tensor_params=[TensorParam("A", is_output=True)],
        scalar_params=[],
        grid_config="simple_2d",
        need_matrixdim=True,
        description="Copy upper triangle to lower triangle of square matrix"
    )
    
    # 3. add_mat: void(torch::Tensor dst, torch::Tensor src, float alpha)
    # 注意：Kaldi 的 add_mat wrapper 需要额外的参数
    print("[3] Generating adapter for: add_mat")
    
    # add_mat 需要自定义的 launcher 调用，因为 Kaldi 的签名是：
    # void cudaF_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, float* dst,
    #                    MatrixDim d, int src_stride, int A_trans)
    add_mat_adapter = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// MatrixDim structure
typedef struct MatrixDim_ {
    int rows;
    int cols;
    int stride;
} MatrixDim;

// Forward declaration
extern "C" void launch_add_mat(dim3 Gr, dim3 Bl, float alpha, const float* src, 
                               float* dst, MatrixDim d, int src_stride, int A_trans);

// dst = alpha * src + dst
void add_mat(torch::Tensor dst, torch::Tensor src, float alpha) {
    // Input validation
    TORCH_CHECK(dst.device().is_cuda(), "dst must be on CUDA");
    TORCH_CHECK(src.device().is_cuda(), "src must be on CUDA");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.sizes() == src.sizes(), "dst and src must have same shape");

    int rows = dst.size(0);
    int cols = dst.size(1);

    // 2D grid configuration
    const int BLOCK_SIZE = 16;
    dim3 Bl(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Gr((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Extract pointers
    float* dst_ptr = dst.data_ptr<float>();
    const float* src_ptr = src.data_ptr<float>();

    // MatrixDim for dst
    MatrixDim d;
    d.rows = rows;
    d.cols = cols;
    d.stride = dst.stride(0);

    int src_stride = src.stride(0);
    int A_trans = 0;  // not transposed

    // Call CUDA launcher
    launch_add_mat(Gr, Bl, alpha, src_ptr, dst_ptr, d, src_stride, A_trans);
    
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_mat", &add_mat, "Matrix addition: dst = alpha * src + dst");
}
'''
    
    # 保存到文件
    from pathlib import Path
    output_dir = Path("./cache/generated_adapters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapters = {
        "copy_low_upp": copy_low_upp_adapter,
        "copy_upp_low": copy_upp_low_adapter,
        "add_mat": add_mat_adapter,
    }
    
    for name, code in adapters.items():
        output_file = output_dir / f"{name}_adapter.cpp"
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"  ✓ Saved: {output_file}")
    
    # 显示一个示例
    print("\n" + "=" * 80)
    print("Example: copy_low_upp_adapter.cpp")
    print("=" * 80)
    print(copy_low_upp_adapter)
    
    print("=" * 80)
    print(f"✓ Generated {len(adapters)} adapters in {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    generate_kaldi_k1_adapters()
