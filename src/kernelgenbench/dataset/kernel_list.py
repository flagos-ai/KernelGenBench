import torch
from torch import FunctionSchema
from enum import Enum
from typing import Dict
from .dataloader import TorchOpsLoader
from logging import getLogger
import os

logger = getLogger(__name__)


def flatten_operator_dict(ops_dict: Dict[str, any], namespace: str = "aten") -> Dict[str, any]:
    """Convert an operator dict to flat structure with key format namespace::op_name

    Args:
        ops_dict: operator dict with keys like 'torch.ops.aten.add'
        namespace: namespace, default 'aten'

    Returns:
        flat dict with keys like 'aten::add'

    Example:
        >>> ops = {'torch.ops.aten.add': torch.ops.aten.add, ...}
        >>> flat_ops = flatten_operator_dict(ops, "aten")
        >>> # flat_ops = {'aten::add': torch.ops.aten.add, ...}
    """
    flat_dict = {}
    for key, value in ops_dict.items():
        # key format: 'torch.ops.aten.add' -> extract 'add'
        op_name = key.split('.')[-1]
        full_name = f"{namespace}::{op_name}"
        flat_dict[full_name] = value
    return flat_dict


class Autograd(Enum):
    enable = True
    disable = False

    @classmethod
    def get_optional_value(cls):
        return [member.name for member in cls]

class DynamicImplInfo:
    def __init__(self):
        self._cache = {}
        self._cache_errors = []
        self.loader = TorchOpsLoader(to_str=False)
        self.namespaces = self.loader.list_namespaces()

    def get(self, api: str, *, namespace: str = "aten"):
        if "::" in api:
            namespace, api = api.split("::", 1)
        if api in self._cache:
            return self._cache[api]
        if api in self._cache_errors:
            return None
        if namespace not in self.namespaces:
            return None
        try:
            schemas = self.loader.get_operator(namespace, api).schemas
            self._cache[api] = self._schemas_to_impl_info(schemas, namespace, api)
            return self._cache[api]
        except Exception as e:
            self._cache_errors.append(api)
            logger.error(f"get impl info for {namespace}.{api} error: {e}")
            return None
        
    def _schemas_to_impl_info(self, schemas, namespace: str, api: str):
        impl_info = []
        for overload_name, schema in schemas.items():
            impl_info.append((f"{api}{'.' + overload_name if overload_name != '' else ''}", Autograd.disable))
        return impl_info

    def __contains__(self, namespace_api_tuple: tuple[str, str] | str):
        if isinstance(namespace_api_tuple, str):
            namespace_api_tuple = ("aten", namespace_api_tuple)
        namespace, api = namespace_api_tuple
        impl_info = self.get(api, namespace=namespace)
        return impl_info is not None
    
    def __getitem__(self, namespace_api_tuple: tuple[str, str] | str):
        if isinstance(namespace_api_tuple, str):
            namespace_api_tuple = ("aten", namespace_api_tuple)
        namespace, api = namespace_api_tuple
        impl_info = self.get(api, namespace=namespace)
        if impl_info is None:
            raise KeyError(f"impl info for {namespace}.{api} not found")
        return impl_info


IMPL_INFO = {
    "abs": [("abs", Autograd.disable)],
    "abs_": [("abs_", Autograd.disable)],
    "add": [("add.Tensor", Autograd.disable)],
    "add_": [("add_.Tensor", Autograd.disable)],
    "addmm": [("addmm", Autograd.disable)],
    "addmv": [("addmv", Autograd.disable)],
    "arange": [
        ("arange.start_step", Autograd.disable),
        ("arange.start", Autograd.disable),
        ("arange", Autograd.disable),
    ],
    "batch_norm": [("batch_norm", Autograd.enable)],
    "bitwise_and": [
        ("bitwise_and.Tensor", Autograd.disable),
        ("bitwise_and.Scalar", Autograd.disable),
        ("bitwise_and.Scalar_Tensor", Autograd.disable),
    ],
    "bitwise_and_": [
        ("bitwise_and_.Tensor_", Autograd.disable),
        ("bitwise_and_.Scalar", Autograd.disable),
    ],
    "bitwise_not": [("bitwise_not", Autograd.disable)],
    "bitwise_not_": [("bitwise_not_", Autograd.disable)],
    "bitwise_or": [
        ("bitwise_or.Tensor", Autograd.disable),
        ("bitwise_or.Scalar", Autograd.disable),
        ("bitwise_or.Scalar_Tensor", Autograd.disable),
    ],
    "bitwise_or_": [
        ("bitwise_or_.Tensor", Autograd.disable),
        ("bitwise_or_.Scalar", Autograd.disable),
    ],
    "bmm": [("bmm", Autograd.disable)],
    "clamp": [
        ("clamp", Autograd.disable),
        ("clamp.Tensor", Autograd.disable),
    ],
    "clamp_": [
        ("clamp_", Autograd.disable),
        ("clamp_.Tensor", Autograd.disable),
    ],
    "cos": [("cos", Autograd.disable)],
    "cos_": [("cos_", Autograd.disable)],
    "pad": [("pad", Autograd.disable)],
    "constant_pad_nd": [("constant_pad_nd", Autograd.disable)],
    "cumsum": [("cumsum", Autograd.disable)],
    "cummin": [("cummin", Autograd.disable)],
    "div": [
        ("div.Tensor", Autograd.disable),
        ("div.Scalar", Autograd.disable),
        ("div.Tensor_mode", Autograd.disable),
        ("div.Scalar_mode", Autograd.disable),
    ],
    "div_": [
        ("div_.Tensor", Autograd.disable),
        ("div_.Scalar", Autograd.disable),
        ("div_.Tensor_mode", Autograd.disable),
        ("div_.Scalar_mode", Autograd.disable),
    ],
    "divide": [
        ("divide.Tensor", Autograd.disable),
        ("divide.Scalar", Autograd.disable),
        ("divide.Tensor_mode", Autograd.disable),
        ("divide.Scalar_mode", Autograd.disable),
    ],
    "divide_": [
        ("divide_.Tensor", Autograd.disable),
        ("divide_.Scalar", Autograd.disable),
        ("divide_.Tensor_mode", Autograd.disable),
        ("divide_.Scalar_mode", Autograd.disable),
    ],
    "true_divide": [
        ("true_divide.Tensor", Autograd.disable),
        ("true_divide.Scalar", Autograd.disable),
    ],
    "true_divide_": [
        ("true_divide_.Tensor", Autograd.disable),
        ("true_divide_.Scalar", Autograd.disable),
    ],
    "floor_divide": [
        ("floor_divide", Autograd.disable),
        ("floor_divide.Scalar", Autograd.disable),
    ],
    "floor_divide_": [
        ("floor_divide_.Tensor", Autograd.disable),
        ("floor_divide_.Scalar", Autograd.disable),
    ],
    "remainder": [
        ("remainder.Tensor", Autograd.disable),
        ("remainder.Scalar", Autograd.disable),
        ("remainder.Scalar_Tensor", Autograd.disable),
    ],
    "remainder_": [
        ("remainder_.Tensor", Autograd.disable),
        ("remainder_.Scalar", Autograd.disable),
    ],
    "native_dropout": [("native_dropout", Autograd.enable)],
    "erf": [("erf", Autograd.disable)],
    "erf_": [("erf_", Autograd.disable)],
    "embedding": [("embedding", Autograd.enable)],
    "eq": [
        ("eq.Tensor", Autograd.disable), 
        ("eq.Scalar", Autograd.disable),
    ],
    "exp": [("exp", Autograd.disable)],
    "exp_": [("exp_", Autograd.disable)],
    "exponential_": [("exponential_", Autograd.disable)],
    "ge": [
        ("ge.Tensor", Autograd.disable),
        ("ge.Scalar", Autograd.disable),
    ],
    "gelu": [("gelu", Autograd.enable)],
    "gelu_": [("gelu_", Autograd.enable)],
    "native_group_norm": [("native_group_norm", Autograd.enable)],
    "_weight_norm_interface": [("_weight_norm_interface", Autograd.enable)],
    "_weight_norm": [("_weight_norm", Autograd.enable)],
    "gt": [
        ("gt.Tensor", Autograd.disable),
        ("gt.Scalar", Autograd.disable),
    ],
    "instance_norm": [("instance_norm", Autograd.enable)],
    "isfinite": [("isfinite", Autograd.disable)],
    "isin": [
        ("isin.Tensor_Tensor", Autograd.disable),
        ("isin.Scalar_Tensor", Autograd.disable),
        ("isin.Tensor_Scalar", Autograd.disable),
    ],
    "isinf": [("isinf", Autograd.disable)],
    "isnan": [("isnan", Autograd.disable)],
    "minimum": [("minimum", Autograd.disable)],
    "maximum": [("maximum", Autograd.disable)],
    "native_layer_norm": [("native_layer_norm", Autograd.enable)],
    "le": [
        ("le.Tensor", Autograd.disable),
        ("le.Scalar", Autograd.disable),
    ],
    "lt": [
        ("lt.Tensor", Autograd.disable),
        ("lt.Scalar", Autograd.disable),
    ],
    "rms_norm": [("rms_norm", Autograd.disable)],
    "rand": [("rand", Autograd.disable)],
    "randn": [("randn", Autograd.disable)],
    "rand_like": [("rand_like", Autograd.disable)],
    "randn_like": [("randn_like", Autograd.disable)],
    "zeros": [("zeros", Autograd.disable)],
    "ones": [("ones", Autograd.disable)],
    "full": [("full", Autograd.disable)],
    "zeros_like": [("zeros_like", Autograd.disable)],
    "ones_like": [("ones_like", Autograd.disable)],
    "full_like": [("full_like", Autograd.disable)],
    "resolve_neg": [("resolve_neg", Autograd.disable)],
    "resolve_conj": [("resolve_conj", Autograd.disable)],
    "normal": [
        ("normal.Tensor_float", Autograd.disable),
        ("normal.float_Tensor", Autograd.disable),
        ("normal.Tensor_Tensor", Autograd.disable),
    ],
    "uniform_": [("uniform_", Autograd.disable)],
    "mean": [
        ("mean", Autograd.disable),
        ("mean.dim", Autograd.disable),
    ],
    "mm": [("mm", Autograd.disable)],
    "mul": [("mul.Tensor", Autograd.disable)],
    "mul_": [("mul_.Tensor", Autograd.disable)],
    "multinomial": [("multinomial", Autograd.disable)],
    "mv": [("mv", Autograd.disable)],
    "ne": [
        ("ne.Tensor", Autograd.disable),
        ("ne.Scalar", Autograd.disable),
    ],
    "neg": [("neg", Autograd.disable)],
    "neg_": [("neg_", Autograd.disable)],
    "pow": [
        ("pow.Scalar", Autograd.disable),
        ("pow.Tensor_Scalar", Autograd.disable),
        ("pow.Tensor_Tensor", Autograd.disable),
    ],
    "pow_": [
        ("pow_.Scalar", Autograd.disable),
        ("pow_.Tensor", Autograd.disable),
    ],
    "reciprocal": [("reciprocal", Autograd.disable)],
    "reciprocal_": [("reciprocal_", Autograd.disable)],
    "relu": [("relu", Autograd.enable)],
    "relu_": [("relu_", Autograd.enable)],
    "rsqrt": [("rsqrt", Autograd.disable)],
    "rsqrt_": [("rsqrt_", Autograd.disable)],
    "sigmoid": [("sigmoid", Autograd.enable)],
    "sigmoid_": [("sigmoid_", Autograd.enable)],
    "silu": [("silu", Autograd.enable)],
    "silu_": [("silu_", Autograd.enable)],
    "sin": [("sin", Autograd.disable)],
    "sin_": [("sin_", Autograd.disable)],
    "softmax": [("softmax.int", Autograd.enable)],
    "sort": [("sort", Autograd.disable)],
    "sub": [("sub.Tensor", Autograd.disable)],
    "sub_": [("sub_.Tensor", Autograd.disable)],
    "tanh": [("tanh", Autograd.enable)],
    "tanh_": [("tanh_", Autograd.enable)],
    "triu": [("triu", Autograd.disable)],
    "var_mean": [("var_mean.correction", Autograd.disable)],
    "linalg_vector_norm": [("linalg_vector_norm", Autograd.disable)],
    "where": [
        ("where.self_out", Autograd.disable),
        ("where.self", Autograd.disable),
        ("where.ScalarSelf", Autograd.disable),
        ("where.ScalarOther", Autograd.disable),
    ],
    "max": [
        ("max", Autograd.disable),
        ("max.dim", Autograd.disable),
    ],
    "min": [
        ("min", Autograd.disable),
        ("min.dim", Autograd.disable),
    ],
    "amax": [("amax", Autograd.disable)],
    "argmax": [("argmax", Autograd.disable)],
    "argmin": [("argmin", Autograd.disable)],
    "prod": [
        ("prod", Autograd.disable),
        ("prod.dim_int", Autograd.disable),
    ],
    "sum": [
        ("sum", Autograd.disable),
        ("sum.dim_IntList", Autograd.disable),
    ],
    "scaled_dot_product_attention": [("scaled_dot_product_attention", Autograd.disable)],
    "all": [
        ("all", Autograd.disable),
        ("all.dim", Autograd.disable),
        ("all.dims", Autograd.disable),
    ],
    "any": [
        ("any", Autograd.disable),
        ("any.dim", Autograd.disable),
        ("any.dims", Autograd.disable),
    ],
    "quantile": [("quantile", Autograd.disable)],
    "log_softmax": [("log_softmax.int", Autograd.enable)],
    "outer": [("outer", Autograd.enable)],
    "cross_entropy": [("cross_entropy_loss", Autograd.enable)],
    "nll_loss_forward": [("nll_loss_forward", Autograd.disable)],
    "nll_loss_backward": [("nll_loss_backward", Autograd.disable)],
    "nll_loss2d_forward": [("nll_loss2d_forward", Autograd.disable)],
    "nll_loss2d_backward": [("nll_loss2d_backward", Autograd.disable)],
    "scatter": [
        ("scatter.src", Autograd.disable),
        ("scatter.reduce", Autograd.disable),
    ],
    "gather": [("gather", Autograd.disable)],
    "gather_backward": [("gather_backward", Autograd.disable)],
    "isclose": [("isclose", Autograd.disable)],
    "allclose": [("allclose", Autograd.disable)],
    "fill": [
        ("fill.Scalar", Autograd.disable),
        ("fill.Tensor", Autograd.disable),
    ],
    "flip": [("flip", Autograd.disable)],
    "slice_scatter": [("slice_scatter", Autograd.disable)],
    "select_scatter": [("select_scatter", Autograd.disable)],
    "index_select": [("index_select", Autograd.disable)],
    "tile": [("tile", Autograd.disable)],
    "masked_fill": [
        ("masked_fill.Tensor", Autograd.disable),
        ("masked_fill.Scalar", Autograd.disable),
    ],
    "masked_fill_": [
        ("masked_fill_.Tensor", Autograd.disable),
        ("masked_fill_.Scalar", Autograd.disable),
    ],
    "_unique2": [("_unique2", Autograd.disable)],
    "_upsample_bicubic2d_aa": [("_upsample_bicubic2d_aa", Autograd.disable)],
    "upsample_nearest2d": [("upsample_nearest2d", Autograd.disable)],
    "nonzero": [("nonzero", Autograd.disable)],
    "repeat": [("repeat", Autograd.disable)],
    "masked_select": [("masked_select", Autograd.disable)],
    "stack": [("stack", Autograd.disable)],
    "hstack": [("hstack", Autograd.disable)],
    "cat": [("cat", Autograd.disable)],
    "repeat_interleave": [
        ("repeat_interleave.self_int", Autograd.disable),
        ("repeat_interleave.Tensor", Autograd.disable),
        ("repeat_interleave.self_Tensor", Autograd.disable),
    ],
    "vstack": [("vstack", Autograd.disable)],
    "randperm": [("randperm", Autograd.disable)],
    "diag": [("diag", Autograd.disable)],
    "diag_embed": [("diag_embed", Autograd.disable)],
    "diagonal_backward": [("diagonal_backward", Autograd.disable)],
    "index_add": [("index_add", Autograd.disable)],
    # "index_fill": [("index_fill", Autograd.disable)],
    "count_nonzero": [("count_nonzero", Autograd.disable)],
    "logical_or": [("logical_or", Autograd.disable)],
    "logical_and": [("logical_and", Autograd.disable)],
    "logical_xor": [("logical_xor", Autograd.disable)],
    "logical_not": [("logical_not", Autograd.disable)],
    "kron": [("kron", Autograd.disable)],
    "elu": [("elu", Autograd.disable)],
    "index_put": [("index_put", Autograd.disable)],
    "log_sigmoid": [("log_sigmoid", Autograd.disable)],
    "vdot": [("vdot", Autograd.disable)],
    "mse_loss": [("mse_loss", Autograd.disable)],
}


VLLM_OPERATOR_NAMES = [
    'allspark_w8a16_gemm', 'apply_repetition_penalties_cuda',
    'awq_gemm', 'awq_marlin_moe_repack',
    'batched_moe_align_block_size', 'concat_and_cache_mla',
    'convert_fp8', 'convert_vertical_slash_indexes',
    'copy_blocks', 'copy_blocks_mla',
    'cp_gather_cache', 'cp_gather_indexer_k_quant_cache',
    'cutlass_pack_scale_fp8', 'cutlass_scaled_mm', 'cutlass_scaled_mm_azp',
    'fused_add_rms_norm', 'fused_qk_norm_rope',
    'gather_and_maybe_dequant_cache',
    'ggml_dequantize', 'ggml_moe_a8', 'ggml_moe_a8_vec',
    'ggml_mul_mat_a8', 'ggml_mul_mat_vec_a8',
    'gptq_gemm', 'gptq_marlin_24_gemm', 'gptq_marlin_gemm',
    'gptq_marlin_moe_repack', 'gptq_shuffle',
    'grouped_topk', 'hadacore_transform',
    'marlin_int4_fp8_preprocess', 'merge_attn_states',
    'moe_align_block_size', 'moe_lora_align_block_size', 'moe_sum',
    'paged_attention_v1', 'paged_attention_v2',
    'permute_cols', 'reshape_and_cache', 'reshape_and_cache_flash',
    'rms_norm', 'rms_norm_dynamic_per_token_quant', 'rms_norm_per_block_quant',
    'rotary_embedding', 'scaled_fp8_quant', 'scaled_int8_quant',
    'selective_scan_fwd', 'shuffle_rows', 'swap_blocks', 'topk_softmax',
]

CUBLAS_OPERATOR_NAMES = [
    'cublasCcopy_v2', 'cublasCdotu_v2',
    'cublasCgemmStridedBatched', 'cublasCgemmStridedBatched_64',
    'cublasCgemm_v2', 'cublasCgemvBatched_64', 'cublasCgemvStridedBatched', 'cublasCgemv_v2',
    'cublasCgeru_v2', 'cublasCsymm_v2', 'cublasCsymv_v2', 'cublasCsyrkEx',
    'cublasDasum_v2', 'cublasDaxpy_v2', 'cublasDcopy_v2',
    'cublasDgemmBatched', 'cublasDgemmStridedBatched', 'cublasDgemmStridedBatched_64',
    'cublasDgemvBatched', 'cublasDgemvStridedBatched', 'cublasDgemv_v2',
    'cublasDsbmv_v2', 'cublasDsyr2_v2', 'cublasDtrsmBatched',
    'cublasHgemmBatched', 'cublasHgemmStridedBatched',
    'cublasSaxpy_v2', 'cublasSdgmm', 'cublasSdot_v2', 'cublasSgeam',
    'cublasSgemmBatched_64', 'cublasSgemmEx', 'cublasSgemmStridedBatched', 'cublasSgemm_v2',
    'cublasSgemvBatched', 'cublasSgemvStridedBatched', 'cublasSger_v2', 'cublasSscal_v2',
    'cublasSsyrk_v2', 'cublasStbmv_v2', 'cublasStrsm_v2', 'cublasStrsv_v2',
    'cublasZdotc_v2', 'cublasZgemmBatched', 'cublasZgemmStridedBatched',
    'cublasZgemvBatched', 'cublasZgemvStridedBatched', 'cublasZgerc_v2',
    'cublasZswap_v2', 'cublasZtrsmBatched',
]

TORCH_OPERATOR_NAMES = [
    '_index_put_impl_', '_local_scalar_dense', '_softmax', '_to_copy',
    'acosh', 'add', 'add_', 'affine_grid_generator', 'amin', 'arange',
    'argmax', 'as_strided', 'asin', 'bernoulli', 'binary_cross_entropy_with_logits',
    'bitwise_not', 'bmm', 'cat', 'clone', 'contiguous', 'copy_', 'cos', 'cosh',
    'cumsum', 'diff', 'div', 'div_', 'embedding', 'empty_strided', 'eq', 'erfc',
    'expand', 'expand_as', 'exponential_', 'fill_', 'floor', 'floor_divide', 'fmax',
    'full', 'gather', 'gt', 'hardsigmoid', 'heaviside', 'huber_loss', 'i0', 'im2col',
    'index', 'index_put_', 'index_select', 'item', 'le', 'linear', 'log10',
    'log_sigmoid_backward', 'logaddexp2', 'logit', 'margin_ranking_loss', 'masked_fill_',
    'matmul', 'mean', 'mish', 'mish_backward', 'mm', 'mul', 'narrow', 'neg',
    'new_empty_strided', 'new_ones', 'ones_like', 'pairwise_distance', 'poisson',
    'polygamma', 'pow', 'prelu', 'reflection_pad1d_backward', 'renorm', 'reshape',
    'resolve_conj', 'resolve_neg', 'rot90', 'rrelu_with_noise', 'rrelu_with_noise_backward',
    'rsqrt', 'rsub', 'scalar_tensor', 'scatter', 'select', 'select_backward', 'sgn',
    'silu', 'sin', 'smooth_l1_loss_backward', 'soft_margin_loss', 'softmax',
    'softplus_backward', 'sort', 'special_entr', 'square', 'stack', 'sub', 'sum', 't',
    'to', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze',
    'upsample_nearest2d_backward', 'zero_', 'zeros', 'zeros_like',
]

KERNELGENBENCH_OPERATOR_NAMES = (
    [f'vllm13::{name}' for name in VLLM_OPERATOR_NAMES] +
    [f'cublas::{name}' for name in CUBLAS_OPERATOR_NAMES] +
    [f'aten::{name}' for name in TORCH_OPERATOR_NAMES]
)


def _load_vllm_operators():
    from .baseline import vllm13
    return {f'vllm13::{name}': getattr(vllm13, name) for name in VLLM_OPERATOR_NAMES}


def _load_cublas_operators():
    from .baseline import cublas
    return {f'cublas::{name}': getattr(cublas, name) for name in CUBLAS_OPERATOR_NAMES}


def _load_torch_operators():
    ops = {}
    for name in TORCH_OPERATOR_NAMES:
        op = getattr(torch.ops.aten, name, None)
        if op is not None:
            ops[f'aten::{name}'] = op
    return ops


def get_vllm_operators():
    return _load_vllm_operators()


def get_cublas_operators():
    return _load_cublas_operators()


def get_aten_operators():
    """Get ATen operators (110 torch operators)."""
    return _load_torch_operators()


def get_kernelgenbench_nocublas_operators():
    """Get KernelGenBench without cuBLAS subset (50 vllm + 110 torch = 160)."""
    ops = {}
    ops.update(_load_vllm_operators())
    ops.update(_load_torch_operators())
    return ops


def get_kernelgenbench_operators():
    """Get all KernelGenBench operators (50 vllm + 50 cublas + 110 torch = 210)."""
    ops = {}
    ops.update(_load_vllm_operators())
    ops.update(_load_cublas_operators())
    ops.update(_load_torch_operators())
    return ops


def is_pytorch_op(api: str, namespace: str = "") -> bool:
    """Return True if api is a registered PyTorch operator in IMPL_INFO."""
    if namespace:
        return False
    return api in IMPL_INFO

# ============ MM Shape-Specific Benchmark ============
# 50 unique shapes x 2 dtypes (f32, f16) = 100 test cases
# Each (shape, dtype) is a standalone benchmark pinned to aten::mm
# Shapes sourced from 2907 HF model training traces (aten.mm.default)

MM_SHAPE_OPERATORS = {
    # label: ((M, K), (K, N), dtype_short)
    # --- Top 8 core shapes (freq > 3000) ---
    "mm_128x768_768x768_f32": ((128, 768), (768, 768), "f32"),
    "mm_128x768_768x768_f16": ((128, 768), (768, 768), "f16"),
    "mm_128x1024_1024x1024_f32": ((128, 1024), (1024, 1024), "f32"),
    "mm_128x1024_1024x1024_f16": ((128, 1024), (1024, 1024), "f16"),
    "mm_128x1024_1024x4096_f32": ((128, 1024), (1024, 4096), "f32"),
    "mm_128x1024_1024x4096_f16": ((128, 1024), (1024, 4096), "f16"),
    "mm_128x4096_4096x4096_f32": ((128, 4096), (4096, 4096), "f32"),
    "mm_128x4096_4096x4096_f16": ((128, 4096), (4096, 4096), "f16"),
    "mm_128x768_768x3072_f32": ((128, 768), (768, 3072), "f32"),
    "mm_128x768_768x3072_f16": ((128, 768), (768, 3072), "f16"),
    "mm_128x3072_3072x768_f32": ((128, 3072), (3072, 768), "f32"),
    "mm_128x3072_3072x768_f16": ((128, 3072), (3072, 768), "f16"),
    "mm_128x4096_4096x1024_f32": ((128, 4096), (4096, 1024), "f32"),
    "mm_128x4096_4096x1024_f16": ((128, 4096), (4096, 1024), "f16"),
    "mm_128x14336_14336x4096_f32": ((128, 14336), (14336, 4096), "f32"),
    "mm_128x14336_14336x4096_f16": ((128, 14336), (14336, 4096), "f16"),
    # --- Shapes #9-#50 (freq 284-4354) ---
    "mm_128x384_384x384_f32": ((128, 384), (384, 384), "f32"),
    "mm_128x384_384x384_f16": ((128, 384), (384, 384), "f16"),
    "mm_128x2304_2304x768_f32": ((128, 2304), (2304, 768), "f32"),
    "mm_128x2304_2304x768_f16": ((128, 2304), (2304, 768), "f16"),
    "mm_128x4096_4096x14336_f32": ((128, 4096), (4096, 14336), "f32"),
    "mm_128x4096_4096x14336_f16": ((128, 4096), (4096, 14336), "f16"),
    "mm_128x768_768x1152_f32": ((128, 768), (768, 1152), "f32"),
    "mm_128x768_768x1152_f16": ((128, 768), (768, 1152), "f16"),
    "mm_128x2048_2048x2048_f32": ((128, 2048), (2048, 2048), "f32"),
    "mm_128x2048_2048x2048_f16": ((128, 2048), (2048, 2048), "f16"),
    "mm_128x5120_5120x5120_f32": ((128, 5120), (5120, 5120), "f32"),
    "mm_128x5120_5120x5120_f16": ((128, 5120), (5120, 5120), "f16"),
    "mm_128x1024_1024x5120_f32": ((128, 1024), (1024, 5120), "f32"),
    "mm_128x1024_1024x5120_f16": ((128, 1024), (1024, 5120), "f16"),
    "mm_128x3072_3072x1024_f32": ((128, 3072), (3072, 1024), "f32"),
    "mm_128x3072_3072x1024_f16": ((128, 3072), (3072, 1024), "f16"),
    "mm_128x384_384x1536_f32": ((128, 384), (384, 1536), "f32"),
    "mm_128x384_384x1536_f16": ((128, 384), (384, 1536), "f16"),
    "mm_128x1536_1536x384_f32": ((128, 1536), (1536, 384), "f32"),
    "mm_128x1536_1536x384_f16": ((128, 1536), (1536, 384), "f16"),
    "mm_768x512_512x768_f32": ((768, 512), (512, 768), "f32"),
    "mm_768x512_512x768_f16": ((768, 512), (512, 768), "f16"),
    "mm_128x1024_1024x2624_f32": ((128, 1024), (1024, 2624), "f32"),
    "mm_128x1024_1024x2624_f16": ((128, 1024), (1024, 2624), "f16"),
    "mm_128x5248_5248x1024_f32": ((128, 5248), (5248, 1024), "f32"),
    "mm_128x5248_5248x1024_f16": ((128, 5248), (5248, 1024), "f16"),
    "mm_1024x512_512x1024_f32": ((1024, 512), (512, 1024), "f32"),
    "mm_1024x512_512x1024_f16": ((1024, 512), (512, 1024), "f16"),
    "mm_128x18944_18944x3584_f32": ((128, 18944), (18944, 3584), "f32"),
    "mm_128x18944_18944x3584_f16": ((128, 18944), (18944, 3584), "f16"),
    "mm_128x3584_3584x3584_f32": ((128, 3584), (3584, 3584), "f32"),
    "mm_128x3584_3584x3584_f16": ((128, 3584), (3584, 3584), "f16"),
    "mm_128x512_512x3584_f32": ((128, 512), (512, 3584), "f32"),
    "mm_128x512_512x3584_f16": ((128, 512), (512, 3584), "f16"),
    "mm_128x14336_14336x5120_f32": ((128, 14336), (14336, 5120), "f32"),
    "mm_128x14336_14336x5120_f16": ((128, 14336), (14336, 5120), "f16"),
    "mm_128x192_192x192_f32": ((128, 192), (192, 192), "f32"),
    "mm_128x192_192x192_f16": ((128, 192), (192, 192), "f16"),
    "mm_128x13824_13824x5120_f32": ((128, 13824), (13824, 5120), "f32"),
    "mm_128x13824_13824x5120_f16": ((128, 13824), (13824, 5120), "f16"),
    "mm_128x2560_2560x2560_f32": ((128, 2560), (2560, 2560), "f32"),
    "mm_128x2560_2560x2560_f16": ((128, 2560), (2560, 2560), "f16"),
    "mm_128x11008_11008x4096_f32": ((128, 11008), (11008, 4096), "f32"),
    "mm_128x11008_11008x4096_f16": ((128, 11008), (11008, 4096), "f16"),
    "mm_128x256_256x2048_f32": ((128, 256), (256, 2048), "f32"),
    "mm_128x256_256x2048_f16": ((128, 256), (256, 2048), "f16"),
    "mm_768x128_128x768_f32": ((768, 128), (128, 768), "f32"),
    "mm_768x128_128x768_f16": ((768, 128), (128, 768), "f16"),
    "mm_128x9216_9216x2304_f32": ((128, 9216), (9216, 2304), "f32"),
    "mm_128x9216_9216x2304_f16": ((128, 9216), (9216, 2304), "f16"),
    "mm_128x1024_1024x2304_f32": ((128, 1024), (1024, 2304), "f32"),
    "mm_128x1024_1024x2304_f16": ((128, 1024), (1024, 2304), "f16"),
    "mm_128x5120_5120x4096_f32": ((128, 5120), (5120, 4096), "f32"),
    "mm_128x5120_5120x4096_f16": ((128, 5120), (5120, 4096), "f16"),
    "mm_128x4096_4096x5120_f32": ((128, 4096), (4096, 5120), "f32"),
    "mm_128x4096_4096x5120_f16": ((128, 4096), (4096, 5120), "f16"),
    "mm_128x512_512x2048_f32": ((128, 512), (512, 2048), "f32"),
    "mm_128x512_512x2048_f16": ((128, 512), (512, 2048), "f16"),
    "mm_128x3584_3584x18944_f32": ((128, 3584), (3584, 18944), "f32"),
    "mm_128x3584_3584x18944_f16": ((128, 3584), (3584, 18944), "f16"),
    "mm_128x5120_5120x14336_f32": ((128, 5120), (5120, 14336), "f32"),
    "mm_128x5120_5120x14336_f16": ((128, 5120), (5120, 14336), "f16"),
    "mm_128x512_512x4096_f32": ((128, 512), (512, 4096), "f32"),
    "mm_128x512_512x4096_f16": ((128, 512), (512, 4096), "f16"),
    "mm_128x6144_6144x768_f32": ((128, 6144), (6144, 768), "f32"),
    "mm_128x6144_6144x768_f16": ((128, 6144), (6144, 768), "f16"),
    "mm_1x768_768x768_f32": ((1, 768), (768, 768), "f32"),
    "mm_1x768_768x768_f16": ((1, 768), (768, 768), "f16"),
    "mm_768x1_1x768_f32": ((768, 1), (1, 768), "f32"),
    "mm_768x1_1x768_f16": ((768, 1), (1, 768), "f16"),
    "mm_128x5120_5120x13824_f32": ((128, 5120), (5120, 13824), "f32"),
    "mm_128x5120_5120x13824_f16": ((128, 5120), (5120, 13824), "f16"),
    "mm_128x1024_1024x3072_f32": ((128, 1024), (1024, 3072), "f32"),
    "mm_128x1024_1024x3072_f16": ((128, 1024), (1024, 3072), "f16"),
    "mm_128x256_256x768_f32": ((128, 256), (256, 768), "f32"),
    "mm_128x256_256x768_f16": ((128, 256), (256, 768), "f16"),
    "mm_128x1152_1152x768_f32": ((128, 1152), (1152, 768), "f32"),
    "mm_128x1152_1152x768_f16": ((128, 1152), (1152, 768), "f16"),
    "mm_128x4096_4096x11008_f32": ((128, 4096), (4096, 11008), "f32"),
    "mm_128x4096_4096x11008_f16": ((128, 4096), (4096, 11008), "f16"),
    "mm_128x11008_11008x2048_f32": ((128, 11008), (11008, 2048), "f32"),
    "mm_128x11008_11008x2048_f16": ((128, 11008), (11008, 2048), "f16"),
    "mm_128x1024_1024x2048_f32": ((128, 1024), (1024, 2048), "f32"),
    "mm_128x1024_1024x2048_f16": ((128, 1024), (1024, 2048), "f16"),
}

MM_SHAPE_OPERATOR_NAMES = list(MM_SHAPE_OPERATORS.keys())


def get_mm_shape_operators():
    """Get MM Shape Benchmark operator dict.

    Returns: {"aten::mm_128x768_768x768_f32": torch.ops.aten.mm, ...}
    All shape variants share the same underlying torch.ops.aten.mm operator.
    """
    mm_op = getattr(torch.ops.aten, "mm", None)
    if mm_op is None:
        raise RuntimeError("torch.ops.aten.mm not found")
    ops = {}
    for name in MM_SHAPE_OPERATOR_NAMES:
        ops[f"aten::{name}"] = mm_op
    return ops


def get_mm_shape_info(label: str):
    """Get shape info: returns ((M, K), (K, N), dtype) or None."""
    return MM_SHAPE_OPERATORS.get(label)


def resolve_op_func_name(op_name: str, namespace: str = "aten") -> str:
    """Resolve a benchmark operator name to the expected function name in code.

    For specialized benchmarks (e.g., MmShape), the benchmark op name
    (mm_128x768_768x768_f32) differs from the actual function name (mm).
    This resolves to the base operator name for such cases.
    For regular operators, returns op_name unchanged.
    """
    if op_name in MM_SHAPE_OPERATORS:
        return "mm"
    return op_name


# if os.environ.get("FLAGBENCH_USE_DYNAMIC_IMPL_INFO", "0") == "1":
#     dynamic_impl_info = DynamicImplInfo()
#     IMPL_INFO = dynamic_impl_info
dynamic_impl_info = DynamicImplInfo()
IMPL_INFO = dynamic_impl_info
    
if __name__ == "__main__":
    op_name_list = list(PYTORCH_OPERATORS.keys())
    dynamic_impl_info = DynamicImplInfo()
