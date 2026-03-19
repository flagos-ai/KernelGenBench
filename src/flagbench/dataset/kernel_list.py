import torch
from torch import FunctionSchema
from enum import Enum
from typing import Dict
from .dataloader import TorchOpsLoader
from logging import getLogger
import os
from .baseline.cupy import (
    caxpy, cdgmm, cdotc, cdotu, cgeam, cgemm, cgemv, cgerc, cgeru, cscal, csyrk,
    dasum, daxpy, ddgmm, ddot, dgeam, dgemm, dgemv, dger, dnrm2, dsbmv, dscal, dsyrk,
    hgemm, sasum, saxpy, sdgmm, sdot, sgeam, sgemm, sgemv, sger, snrm2, ssbmv, sscal, ssyrk,
    zaxpy, zdgmm, zdotc, zdotu, zgeam, zgemm, zgemv, zgerc, zgeru, zscal, zsyrk
)

logger = getLogger(__name__)


def flatten_operator_dict(ops_dict: Dict[str, any], namespace: str = "aten") -> Dict[str, any]:
    """将算子字典转换为扁平结构，key格式为 namespace::op_name

    用于将预定义算子字典（如 V2_OPERATORS）从原始格式转换为统一的扁平格式。

    Args:
        ops_dict: 原始算子字典，key格式为 'torch.ops.aten.add'
        namespace: 命名空间，默认为 'aten'

    Returns:
        扁平化字典，key格式为 'aten::add'

    Example:
        >>> V2_OPERATORS = {'torch.ops.aten.add': torch.ops.aten.add, ...}
        >>> flat_ops = flatten_operator_dict(V2_OPERATORS, "aten")
        >>> # flat_ops = {'aten::add': torch.ops.aten.add, ...}
    """
    flat_dict = {}
    for key, value in ops_dict.items():
        # key 格式: 'torch.ops.aten.add' -> 提取 'add'
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
        assert namespace in self.namespaces, f"namespace {namespace} not found"
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


PYTORCH_OPERATORS = {
    'torch.abs': torch.abs,
    'torch.abs_': torch.abs_,
    'torch.add': torch.add,
    'torch.Tensor.add_': torch.Tensor.add_,
    'torch.addmm': torch.addmm,
    'torch.addmv': torch.addmv,
    'torch.all': torch.all,
    'torch.allclose': torch.allclose,
    'torch.amax': torch.amax,
    'torch.angle': torch.angle,
    'torch.any': torch.any,
    'torch.arange': torch.arange,
    'torch.argmax': torch.argmax,
    'torch.argmin': torch.argmin,
    'torch.batch_norm': torch.batch_norm,
    'torch.bitwise_and': torch.bitwise_and,
    'torch.Tensor.bitwise_and_': torch.Tensor.bitwise_and_,
    'torch.bitwise_not': torch.bitwise_not,
    'torch.Tensor.bitwise_not_': torch.Tensor.bitwise_not_,
    'torch.bitwise_or': torch.bitwise_or,
    'torch.Tensor.bitwise_or_': torch.Tensor.bitwise_or_,
    'torch.bmm': torch.bmm,
    'torch.cat': torch.cat,
    'torch.clamp': torch.clamp,
    'torch.clamp_': torch.clamp_,
    'torch.Tensor.contiguous': torch.Tensor.contiguous,
    'torch.conv1d': torch.conv1d,
    'torch.conv2d': torch.conv2d,
    'torch.cos': torch.cos,
    'torch.cos_': torch.cos_,
    'torch.count_nonzero': torch.count_nonzero,
    'torch.nn.functional.cross_entropy': torch.nn.functional.cross_entropy,
    'torch.cummax': torch.cummax,
    'torch.cummin': torch.cummin,
    # 'torch.cumsum': torch.cumsum,
    'torch.diag': torch.diag,
    'torch.diag_embed': torch.diag_embed,
    'torch.diagonal': torch.diagonal,
    'torch.div': torch.div,
    'torch.Tensor.div_': torch.Tensor.div_,
    # 'torch.dot': torch.dot,
    'torch.dropout': torch.dropout,
    'torch.nn.functional.elu': torch.nn.functional.elu,
    # 'torch.nn.functional.elu_': torch.nn.functional.elu_,
    'torch.embedding': torch.embedding,
    'torch.eq': torch.eq,
    'torch.erf': torch.erf,
    'torch.erf_': torch.erf_,
    'torch.exp': torch.exp,
    'torch.exp_': torch.exp_,
    'torch.Tensor.exponential_': torch.Tensor.exponential_,
    'torch.eye': torch.eye,
    'torch.fill': torch.fill,
    'torch.fill_': torch.fill_,
    'torch.flip': torch.flip,
    'torch.floor_divide': torch.floor_divide,
    'torch.Tensor.floor_divide_': torch.Tensor.floor_divide_,
    'torch.full': torch.full,
    'torch.full_like': torch.full_like,
    'torch.gather': torch.gather,
    'torch.ge': torch.ge,
    'torch.nn.functional.gelu': torch.nn.functional.gelu,
    'torch._C._nn.gelu_': torch._C._nn.gelu_,
    'torch.nn.functional.glu': torch.nn.functional.glu,
    'torch.group_norm': torch.group_norm,
    'torch.gt': torch.gt,
    'torch.hstack': torch.hstack,
    'torch.ops.aten.index': torch.ops.aten.index,
    'torch.index_add': torch.index_add,
    'torch.index_put': torch.index_put,
    'torch.index_put_': torch.index_put_,
    'torch.index_select': torch.index_select,
    # 'torch.instance_norm': torch.instance_norm,
    'torch.isclose': torch.isclose,
    'torch.isfinite': torch.isfinite,
    'torch.isin': torch.isin,
    'torch.isinf': torch.isinf,
    'torch.isnan': torch.isnan,
    'torch.kron': torch.kron,
    'torch.layer_norm': torch.layer_norm,
    'torch.le': torch.le,
    'torch.lerp': torch.lerp,
    'torch.Tensor.lerp_': torch.Tensor.lerp_,
    'torch.linspace': torch.linspace,
    'torch.log': torch.log,
    'torch.nn.functional.logsigmoid': torch.nn.functional.logsigmoid,
    'torch.log_softmax': torch.log_softmax,
    'torch.logical_and': torch.logical_and,
    'torch.logical_not': torch.logical_not,
    'torch.logical_or': torch.logical_or,
    'torch.logical_xor': torch.logical_xor,
    'torch.lt': torch.lt,
    'torch.masked_fill': torch.masked_fill,
    'torch.Tensor.masked_fill_': torch.Tensor.masked_fill_,
    'torch.masked_select': torch.masked_select,
    'torch.max': torch.max,
    'torch.maximum': torch.maximum,
    'torch.mean': torch.mean,
    'torch.min': torch.min,
    'torch.minimum': torch.minimum,
    'torch.mm': torch.mm,
    'torch.nn.functional.mse_loss': torch.nn.functional.mse_loss,
    'torch.mul': torch.mul,
    'torch.Tensor.mul_': torch.Tensor.mul_,
    # 'torch.multinomial': torch.multinomial,
    'torch.mv': torch.mv,
    'torch.nan_to_num': torch.nan_to_num,
    'torch.ne': torch.ne,
    'torch.neg': torch.neg,
    'torch.neg_': torch.neg_,
    'torch.nn.functional.nll_loss': torch.nn.functional.nll_loss,
    'torch.nonzero': torch.nonzero,
    'torch.normal': torch.normal,
    'torch.ones': torch.ones,
    'torch.ones_like': torch.ones_like,
    'torch.outer': torch.outer,
    'torch.nn.functional.pad': torch.nn.functional.pad,
    'torch.polar': torch.polar,
    'torch.pow': torch.pow,
    'torch.Tensor.pow_': torch.Tensor.pow_,
    'torch.prod': torch.prod,
    'torch.quantile': torch.quantile,
    'torch.rand': torch.rand,
    'torch.rand_like': torch.rand_like,
    'torch.randn': torch.randn,
    'torch.randn_like': torch.randn_like,
    'torch.randperm': torch.randperm,
    'torch.reciprocal': torch.reciprocal,
    'torch.reciprocal_': torch.reciprocal_,
    'torch.relu': torch.relu,
    'torch.relu_': torch.relu_,
    'torch.remainder': torch.remainder,
    'torch.Tensor.remainder_': torch.Tensor.remainder_,
    'torch.Tensor.repeat': torch.Tensor.repeat,
    'torch.repeat_interleave': torch.repeat_interleave,
    'torch.resolve_conj': torch.resolve_conj,
    'torch.resolve_neg': torch.resolve_neg,
    'torch.rms_norm': torch.rms_norm,
    'torch.rsqrt': torch.rsqrt,
    'torch.rsqrt_': torch.rsqrt_,
    'torch.nn.functional.scaled_dot_product_attention': torch.nn.functional.scaled_dot_product_attention,
    'torch.scatter': torch.scatter,
    'torch.Tensor.scatter_': torch.Tensor.scatter_,
    'torch.select_scatter': torch.select_scatter,
    'torch.sigmoid': torch.sigmoid,
    'torch.sigmoid_': torch.sigmoid_,
    'torch.nn.functional.silu': torch.nn.functional.silu,
    'torch._C._nn.silu_': torch._C._nn.silu_,
    'torch.sin': torch.sin,
    'torch.sin_': torch.sin_,
    'torch.slice_scatter': torch.slice_scatter,
    'torch.softmax': torch.softmax,
    'torch.sort': torch.sort,
    'torch.stack': torch.stack,
    'torch.sub': torch.sub,
    'torch.Tensor.sub_': torch.Tensor.sub_,
    'torch.sum': torch.sum,
    'torch.tanh': torch.tanh,
    'torch.tanh_': torch.tanh_,
    'torch.threshold': torch.threshold,
    'torch.tile': torch.tile,
    'torch.Tensor.to': torch.Tensor.to,
    'torch.topk': torch.topk,
    'torch.triu': torch.triu,
    'torch.Tensor.uniform_': torch.Tensor.uniform_,
    'torch.unique': torch.unique,
    'torch.nn.functional.upsample': torch.nn.functional.upsample,
    'torch.var_mean': torch.var_mean,
    'torch.vdot': torch.vdot,
    'torch.linalg.vector_norm': torch.linalg.vector_norm,
    'torch.vstack': torch.vstack,
    'torch._weight_norm': torch._weight_norm,
    'torch.where': torch.where,
    'torch.zeros': torch.zeros,
    'torch.zeros_like': torch.zeros_like,
    'torch.true_divide': torch.true_divide,
    'torch.Tensor.true_divide_': torch.Tensor.true_divide_,
    'torch.divide': torch.divide,
    'torch.Tensor.divide_': torch.Tensor.divide_, 
    'torch.index_fill': torch.index_fill, 
}

# Selected 40 operators for benchmark library
# These operators are selected based on correctness test results from gpt-5.1 evaluation
# and have corresponding performance tests
V1_OPERATORS = {
    'torch.abs': torch.abs,
    'torch.all': torch.all,
    'torch.allclose': torch.allclose,
    'torch.amax': torch.amax,
    'torch.any': torch.any,
    'torch.arange': torch.arange,
    'torch.argmax': torch.argmax,
    'torch.argmin': torch.argmin,
    'torch.bitwise_and': torch.bitwise_and,
    'torch.bitwise_not': torch.bitwise_not,
    'torch.bitwise_or': torch.bitwise_or,
    'torch.cos': torch.cos,
    'torch.count_nonzero': torch.count_nonzero,
    'torch.diag': torch.diag,
    'torch.diag_embed': torch.diag_embed,
    'torch.div': torch.div,
    'torch.embedding': torch.embedding,
    'torch.eq': torch.eq,
    'torch.fill': torch.fill,
    'torch.floor_divide': torch.floor_divide,
    'torch.full': torch.full,
    'torch.full_like': torch.full_like,
    'torch.gather': torch.gather,
    'torch.ge': torch.ge,
    'torch.gt': torch.gt,
    'torch.index_add': torch.index_add,
    'torch.isfinite': torch.isfinite,
    'torch.isinf': torch.isinf,
    'torch.isnan': torch.isnan,
    'torch.kron': torch.kron,
    'torch.mean': torch.mean,
    'torch.mul': torch.mul,
    'torch.nn.functional.scaled_dot_product_attention': torch.nn.functional.scaled_dot_product_attention,
    'torch.ones': torch.ones,
    'torch.rand': torch.rand,
    'torch.relu': torch.relu,
    'torch.resolve_conj': torch.resolve_conj,
    'torch.tanh': torch.tanh,
    'torch.vdot': torch.vdot,
    'torch.zeros_like': torch.zeros_like,
}

# Non-FlagGems operators: 10 operators from log_9 result.json
# These operators are not in FlagGems but have test functions generated
NON_FLAGGEMS_OPERATORS = {
    'torch.ops.aten.log_normal': torch.ops.aten.log_normal,
    'torch.ops.aten.bernoulli': torch.ops.aten.bernoulli,
    'torch.ops.aten.unfold_backward': torch.ops.aten.unfold_backward,
    'torch.ops.aten.logit_backward': torch.ops.aten.logit_backward,
    'torch.ops.aten.convolution': torch.ops.aten.convolution,
    'torch.ops.aten.linalg_cross': torch.ops.aten.linalg_cross,
    'torch.ops.aten.avg_pool3d': torch.ops.aten.avg_pool3d,
    'torch.ops.aten.round': torch.ops.aten.round,
    'torch.ops.aten.baddbmm': torch.ops.aten.baddbmm,
    'torch.ops.aten.addbmm': torch.ops.aten.addbmm,
}

# V2 operators: 50 operators from sampled_from_passed_ops.json
# These operators have test functions generated and extracted to test_v2_ops.py
V2_OPERATORS = {
    # Backward operators
    'torch.ops.aten.log_sigmoid_backward': torch.ops.aten.log_sigmoid_backward,
    'torch.ops.aten.mish_backward': torch.ops.aten.mish_backward,
    'torch.ops.aten.reflection_pad1d_backward': torch.ops.aten.reflection_pad1d_backward,
    'torch.ops.aten.rrelu_with_noise_backward': torch.ops.aten.rrelu_with_noise_backward,
    'torch.ops.aten.select_backward': torch.ops.aten.select_backward,
    'torch.ops.aten.smooth_l1_loss_backward': torch.ops.aten.smooth_l1_loss_backward,
    'torch.ops.aten.softplus_backward': torch.ops.aten.softplus_backward,
    'torch.ops.aten.upsample_nearest2d_backward': torch.ops.aten.upsample_nearest2d_backward,
    # Activation functions
    'torch.ops.aten.erfc': torch.ops.aten.erfc,
    'torch.ops.aten.hardsigmoid': torch.ops.aten.hardsigmoid,
    'torch.ops.aten.heaviside': torch.ops.aten.heaviside,
    'torch.ops.aten.log10': torch.ops.aten.log10,
    'torch.ops.aten.logit': torch.ops.aten.logit,
    'torch.ops.aten.mish': torch.ops.aten.mish,
    'torch.ops.aten.prelu': torch.ops.aten.prelu,
    'torch.ops.aten.rrelu_with_noise': torch.ops.aten.rrelu_with_noise,
    'torch.ops.aten.square': torch.ops.aten.square,
    # Tensor creation and manipulation
    'torch.ops.aten.affine_grid_generator': torch.ops.aten.affine_grid_generator,
    'torch.ops.aten.bernoulli': torch.ops.aten.bernoulli,
    'torch.ops.aten.empty_strided': torch.ops.aten.empty_strided,
    'torch.ops.aten.new_empty_strided': torch.ops.aten.new_empty_strided,
    'torch.ops.aten.new_ones': torch.ops.aten.new_ones,
    'torch.ops.aten.poisson': torch.ops.aten.poisson,
    'torch.ops.aten.scalar_tensor': torch.ops.aten.scalar_tensor,
    # Math operations
    'torch.ops.aten.acosh': torch.ops.aten.acosh,
    'torch.ops.aten.asin': torch.ops.aten.asin,
    'torch.ops.aten.cosh': torch.ops.aten.cosh,
    'torch.ops.aten.floor': torch.ops.aten.floor,
    'torch.ops.aten.i0': torch.ops.aten.i0,
    'torch.ops.aten.polygamma': torch.ops.aten.polygamma,
    'torch.ops.aten.rsub': torch.ops.aten.rsub,
    'torch.ops.aten.sgn': torch.ops.aten.sgn,
    'torch.ops.aten.special_entr': torch.ops.aten.special_entr,
    # Reduction and comparison operations
    'torch.ops.aten.amin': torch.ops.aten.amin,
    'torch.ops.aten.binary_cross_entropy_with_logits': torch.ops.aten.binary_cross_entropy_with_logits,
    'torch.ops.aten.fmax': torch.ops.aten.fmax,
    'torch.ops.aten.huber_loss': torch.ops.aten.huber_loss,
    'torch.ops.aten.logaddexp2': torch.ops.aten.logaddexp2,
    'torch.ops.aten.margin_ranking_loss': torch.ops.aten.margin_ranking_loss,
    'torch.ops.aten.pairwise_distance': torch.ops.aten.pairwise_distance,
    'torch.ops.aten.renorm': torch.ops.aten.renorm,
    'torch.ops.aten.soft_margin_loss': torch.ops.aten.soft_margin_loss,
    # Tensor shape operations
    'torch.ops.aten.as_strided': torch.ops.aten.as_strided,
    'torch.ops.aten.im2col': torch.ops.aten.im2col,
    'torch.ops.aten.reshape': torch.ops.aten.reshape,
    'torch.ops.aten.rot90': torch.ops.aten.rot90,
    'torch.ops.aten.t': torch.ops.aten.t,
    'torch.ops.aten.unsafe_split': torch.ops.aten.unsafe_split,
    'torch.ops.aten.unsafe_split_with_sizes': torch.ops.aten.unsafe_split_with_sizes,
    'torch.ops.aten.unsqueeze': torch.ops.aten.unsqueeze,
}


# Qwen next operators
# aten::_flash_attention_backward
# aten::_flash_attention_forward
# aten::_index_put_impl_
# aten::_local_scalar_dense
# aten::_scaled_dot_product_flash_attention
# aten::_scaled_dot_product_flash_attention_backward
# aten::_softmax
# aten::_to_copy
# aten::add
# aten::add_
# aten::arange
# aten::argmax
# aten::bitwise_not
# aten::bmm
# aten::cat
# aten::clone
# aten::contiguous
# aten::copy_
# aten::cos
# aten::cumsum
# aten::diff
# aten::div
# aten::div_
# aten::embedding
# aten::embedding_backward
# aten::embedding_dense_backward
# aten::eq
# aten::expand
# aten::expand_as
# aten::exponential_
# aten::fill_
# aten::floor_divide
# aten::full
# aten::gather
# aten::gt
# aten::index
# aten::index_put_
# aten::index_select
# aten::item
# aten::le
# aten::linear
# aten::masked_fill_
# aten::matmul
# aten::mean
# aten::mm
# aten::mul
# aten::narrow
# aten::neg
# aten::ones_like
# aten::pow
# aten::resolve_conj
# aten::resolve_neg
# aten::rsqrt
# aten::rsub
# aten::scaled_dot_product_attention
# aten::scatter
# aten::select
# aten::silu
# aten::silu_backward
# aten::sin
# aten::softmax
# aten::sort
# aten::stack
# aten::sub
# aten::sum
# aten::to
# aten::zero_
# aten::zeros
# aten::zeros_like

# Qwen next operators
QWEN_NEXT_OPERATORS = {
    # 'torch.ops.aten._flash_attention_backward': torch.ops.aten._flash_attention_backward,
    # 'torch.ops.aten._flash_attention_forward': torch.ops.aten._flash_attention_forward,
    'torch.ops.aten._index_put_impl_': torch.ops.aten._index_put_impl_,
    'torch.ops.aten._local_scalar_dense': torch.ops.aten._local_scalar_dense,
    # 'torch.ops.aten._scaled_dot_product_flash_attention': torch.ops.aten._scaled_dot_product_flash_attention,
    # 'torch.ops.aten._scaled_dot_product_flash_attention_backward': torch.ops.aten._scaled_dot_product_flash_attention_backward,
    'torch.ops.aten._softmax': torch.ops.aten._softmax,
    'torch.ops.aten._to_copy': torch.ops.aten._to_copy,
    'torch.ops.aten.add': torch.ops.aten.add,
    'torch.ops.aten.add_': torch.ops.aten.add_,
    'torch.ops.aten.arange': torch.ops.aten.arange,
    'torch.ops.aten.argmax': torch.ops.aten.argmax,
    'torch.ops.aten.bitwise_not': torch.ops.aten.bitwise_not,
    'torch.ops.aten.bmm': torch.ops.aten.bmm,
    'torch.ops.aten.cat': torch.ops.aten.cat,
    'torch.ops.aten.clone': torch.ops.aten.clone,
    'torch.ops.aten.contiguous': torch.ops.aten.contiguous,
    'torch.ops.aten.copy_': torch.ops.aten.copy_,
    'torch.ops.aten.cos': torch.ops.aten.cos,
    'torch.ops.aten.cumsum': torch.ops.aten.cumsum,
    'torch.ops.aten.diff': torch.ops.aten.diff,
    'torch.ops.aten.div': torch.ops.aten.div,
    'torch.ops.aten.div_': torch.ops.aten.div_,
    'torch.ops.aten.embedding': torch.ops.aten.embedding,
    # 'torch.ops.aten.embedding_backward': torch.ops.aten.embedding_backward,
    # 'torch.ops.aten.embedding_dense_backward': torch.ops.aten.embedding_dense_backward,
    'torch.ops.aten.eq': torch.ops.aten.eq,
    'torch.ops.aten.expand': torch.ops.aten.expand,
    'torch.ops.aten.expand_as': torch.ops.aten.expand_as,
    'torch.ops.aten.exponential_': torch.ops.aten.exponential_,
    'torch.ops.aten.fill_': torch.ops.aten.fill_,
    'torch.ops.aten.floor_divide': torch.ops.aten.floor_divide,
    'torch.ops.aten.full': torch.ops.aten.full,
    'torch.ops.aten.gather': torch.ops.aten.gather,
    'torch.ops.aten.gt': torch.ops.aten.gt,
    'torch.ops.aten.index': torch.ops.aten.index,
    'torch.ops.aten.index_put_': torch.ops.aten.index_put_,
    'torch.ops.aten.index_select': torch.ops.aten.index_select,
    'torch.ops.aten.item': torch.ops.aten.item,
    'torch.ops.aten.le': torch.ops.aten.le,
    'torch.ops.aten.linear': torch.ops.aten.linear,
    'torch.ops.aten.masked_fill_': torch.ops.aten.masked_fill_,
    'torch.ops.aten.matmul': torch.ops.aten.matmul,
    'torch.ops.aten.mean': torch.ops.aten.mean,
    'torch.ops.aten.mm': torch.ops.aten.mm,
    'torch.ops.aten.mul': torch.ops.aten.mul,
    'torch.ops.aten.narrow': torch.ops.aten.narrow,
    'torch.ops.aten.neg': torch.ops.aten.neg,
    'torch.ops.aten.ones_like': torch.ops.aten.ones_like,
    'torch.ops.aten.pow': torch.ops.aten.pow,
    'torch.ops.aten.resolve_conj': torch.ops.aten.resolve_conj,
    'torch.ops.aten.resolve_neg': torch.ops.aten.resolve_neg,
    'torch.ops.aten.rsqrt': torch.ops.aten.rsqrt,
    'torch.ops.aten.rsub': torch.ops.aten.rsub,
    # 'torch.ops.aten.scaled_dot_product_attention': torch.ops.aten.scaled_dot_product_attention,
    'torch.ops.aten.scatter': torch.ops.aten.scatter,
    'torch.ops.aten.select': torch.ops.aten.select,
    'torch.ops.aten.silu': torch.ops.aten.silu,
    # 'torch.ops.aten.silu_backward': torch.ops.aten.silu_backward,
    'torch.ops.aten.sin': torch.ops.aten.sin,
    'torch.ops.aten.softmax': torch.ops.aten.softmax,
    'torch.ops.aten.sort': torch.ops.aten.sort,
    'torch.ops.aten.stack': torch.ops.aten.stack,
    'torch.ops.aten.sub': torch.ops.aten.sub,
    'torch.ops.aten.sum': torch.ops.aten.sum,
    'torch.ops.aten.to': torch.ops.aten.to,
    'torch.ops.aten.zero_': torch.ops.aten.zero_,
    'torch.ops.aten.zeros': torch.ops.aten.zeros,
    'torch.ops.aten.zeros_like': torch.ops.aten.zeros_like,
}

V2_1_OPERATORS = {
    'torch.ops.aten._index_put_impl_': torch.ops.aten._index_put_impl_,
    'torch.ops.aten._local_scalar_dense': torch.ops.aten._local_scalar_dense,
    'torch.ops.aten._softmax': torch.ops.aten._softmax,
    'torch.ops.aten._to_copy': torch.ops.aten._to_copy,
    'torch.ops.aten.acosh': torch.ops.aten.acosh,
    'torch.ops.aten.add': torch.ops.aten.add,
    'torch.ops.aten.add_': torch.ops.aten.add_,
    'torch.ops.aten.affine_grid_generator': torch.ops.aten.affine_grid_generator,
    'torch.ops.aten.amin': torch.ops.aten.amin,
    'torch.ops.aten.arange': torch.ops.aten.arange,
    'torch.ops.aten.argmax': torch.ops.aten.argmax,
    'torch.ops.aten.as_strided': torch.ops.aten.as_strided,
    'torch.ops.aten.asin': torch.ops.aten.asin,
    'torch.ops.aten.bernoulli': torch.ops.aten.bernoulli,
    'torch.ops.aten.binary_cross_entropy_with_logits': torch.ops.aten.binary_cross_entropy_with_logits,
    'torch.ops.aten.bitwise_not': torch.ops.aten.bitwise_not,
    'torch.ops.aten.bmm': torch.ops.aten.bmm,
    'torch.ops.aten.cat': torch.ops.aten.cat,
    'torch.ops.aten.clone': torch.ops.aten.clone,
    'torch.ops.aten.contiguous': torch.ops.aten.contiguous,
    'torch.ops.aten.copy_': torch.ops.aten.copy_,
    'torch.ops.aten.cos': torch.ops.aten.cos,
    'torch.ops.aten.cosh': torch.ops.aten.cosh,
    'torch.ops.aten.cumsum': torch.ops.aten.cumsum,
    'torch.ops.aten.diff': torch.ops.aten.diff,
    'torch.ops.aten.div': torch.ops.aten.div,
    'torch.ops.aten.div_': torch.ops.aten.div_,
    'torch.ops.aten.embedding': torch.ops.aten.embedding,
    'torch.ops.aten.empty_strided': torch.ops.aten.empty_strided,
    'torch.ops.aten.eq': torch.ops.aten.eq,
    'torch.ops.aten.erfc': torch.ops.aten.erfc,
    'torch.ops.aten.expand': torch.ops.aten.expand,
    'torch.ops.aten.expand_as': torch.ops.aten.expand_as,
    'torch.ops.aten.exponential_': torch.ops.aten.exponential_,
    'torch.ops.aten.fill_': torch.ops.aten.fill_,
    'torch.ops.aten.floor': torch.ops.aten.floor,
    'torch.ops.aten.floor_divide': torch.ops.aten.floor_divide,
    'torch.ops.aten.fmax': torch.ops.aten.fmax,
    'torch.ops.aten.full': torch.ops.aten.full,
    'torch.ops.aten.gather': torch.ops.aten.gather,
    'torch.ops.aten.gt': torch.ops.aten.gt,
    'torch.ops.aten.hardsigmoid': torch.ops.aten.hardsigmoid,
    'torch.ops.aten.heaviside': torch.ops.aten.heaviside,
    'torch.ops.aten.huber_loss': torch.ops.aten.huber_loss,
    'torch.ops.aten.i0': torch.ops.aten.i0,
    'torch.ops.aten.im2col': torch.ops.aten.im2col,
    'torch.ops.aten.index': torch.ops.aten.index,
    'torch.ops.aten.index_put_': torch.ops.aten.index_put_,
    'torch.ops.aten.index_select': torch.ops.aten.index_select,
    'torch.ops.aten.item': torch.ops.aten.item,
    'torch.ops.aten.le': torch.ops.aten.le,
    'torch.ops.aten.linear': torch.ops.aten.linear,
    'torch.ops.aten.log10': torch.ops.aten.log10,
    'torch.ops.aten.log_sigmoid_backward': torch.ops.aten.log_sigmoid_backward,
    'torch.ops.aten.logaddexp2': torch.ops.aten.logaddexp2,
    'torch.ops.aten.logit': torch.ops.aten.logit,
    'torch.ops.aten.margin_ranking_loss': torch.ops.aten.margin_ranking_loss,
    'torch.ops.aten.masked_fill_': torch.ops.aten.masked_fill_,
    'torch.ops.aten.matmul': torch.ops.aten.matmul,
    'torch.ops.aten.mean': torch.ops.aten.mean,
    'torch.ops.aten.mish': torch.ops.aten.mish,
    'torch.ops.aten.mish_backward': torch.ops.aten.mish_backward,
    'torch.ops.aten.mm': torch.ops.aten.mm,
    'torch.ops.aten.mul': torch.ops.aten.mul,
    'torch.ops.aten.narrow': torch.ops.aten.narrow,
    'torch.ops.aten.neg': torch.ops.aten.neg,
    'torch.ops.aten.new_empty_strided': torch.ops.aten.new_empty_strided,
    'torch.ops.aten.new_ones': torch.ops.aten.new_ones,
    'torch.ops.aten.ones_like': torch.ops.aten.ones_like,
    'torch.ops.aten.pairwise_distance': torch.ops.aten.pairwise_distance,
    'torch.ops.aten.poisson': torch.ops.aten.poisson,
    'torch.ops.aten.polygamma': torch.ops.aten.polygamma,
    'torch.ops.aten.pow': torch.ops.aten.pow,
    'torch.ops.aten.prelu': torch.ops.aten.prelu,
    'torch.ops.aten.reflection_pad1d_backward': torch.ops.aten.reflection_pad1d_backward,
    'torch.ops.aten.renorm': torch.ops.aten.renorm,
    'torch.ops.aten.reshape': torch.ops.aten.reshape,
    'torch.ops.aten.resolve_conj': torch.ops.aten.resolve_conj,
    'torch.ops.aten.resolve_neg': torch.ops.aten.resolve_neg,
    'torch.ops.aten.rot90': torch.ops.aten.rot90,
    'torch.ops.aten.rrelu_with_noise': torch.ops.aten.rrelu_with_noise,
    'torch.ops.aten.rrelu_with_noise_backward': torch.ops.aten.rrelu_with_noise_backward,
    'torch.ops.aten.rsqrt': torch.ops.aten.rsqrt,
    'torch.ops.aten.rsub': torch.ops.aten.rsub,
    'torch.ops.aten.scalar_tensor': torch.ops.aten.scalar_tensor,
    'torch.ops.aten.scatter': torch.ops.aten.scatter,
    'torch.ops.aten.select': torch.ops.aten.select,
    'torch.ops.aten.select_backward': torch.ops.aten.select_backward,
    'torch.ops.aten.sgn': torch.ops.aten.sgn,
    'torch.ops.aten.silu': torch.ops.aten.silu,
    'torch.ops.aten.sin': torch.ops.aten.sin,
    'torch.ops.aten.smooth_l1_loss_backward': torch.ops.aten.smooth_l1_loss_backward,
    'torch.ops.aten.soft_margin_loss': torch.ops.aten.soft_margin_loss,
    'torch.ops.aten.softmax': torch.ops.aten.softmax,
    'torch.ops.aten.softplus_backward': torch.ops.aten.softplus_backward,
    'torch.ops.aten.sort': torch.ops.aten.sort,
    'torch.ops.aten.special_entr': torch.ops.aten.special_entr,
    'torch.ops.aten.square': torch.ops.aten.square,
    'torch.ops.aten.stack': torch.ops.aten.stack,
    'torch.ops.aten.sub': torch.ops.aten.sub,
    'torch.ops.aten.sum': torch.ops.aten.sum,
    'torch.ops.aten.t': torch.ops.aten.t,
    'torch.ops.aten.to': torch.ops.aten.to,
    'torch.ops.aten.unsafe_split': torch.ops.aten.unsafe_split,
    'torch.ops.aten.unsafe_split_with_sizes': torch.ops.aten.unsafe_split_with_sizes,
    'torch.ops.aten.unsqueeze': torch.ops.aten.unsqueeze,
    'torch.ops.aten.upsample_nearest2d_backward': torch.ops.aten.upsample_nearest2d_backward,
    'torch.ops.aten.zero_': torch.ops.aten.zero_,
    'torch.ops.aten.zeros': torch.ops.aten.zeros,
    'torch.ops.aten.zeros_like': torch.ops.aten.zeros_like,
}

def is_pytorch_op(name: str, *, namespace: str = "aten") -> bool:
    """
    判断算子是否是 PyTorch 算子

    Args:
        name: 算子名称（可以带 framework 前缀，如 "abs"、"aten::abs"、"vllm13::rms_norm"）

    Returns:
        True 如果是 PyTorch 算子，False 否则
    """
    # 如果带 framework:: 前缀，直接判断是否为 aten
    if "::" in name:
        framework = name.split("::")[0]
        return framework == "aten"
    try:
        return IMPL_INFO.get(name, namespace=namespace) is not None
    except (AssertionError, KeyError):
        # 如果 namespace 不存在（如 cupy），说明不是 PyTorch 算子
        return False


CUPY_OPERATORS = {
    'cupy::caxpy': caxpy,
    'cupy::cdgmm': cdgmm,
    'cupy::cdotc': cdotc,
    'cupy::cdotu': cdotu,
    'cupy::cgeam': cgeam,
    'cupy::cgemm': cgemm,
    'cupy::cgemv': cgemv,
    'cupy::cgerc': cgerc,
    'cupy::cgeru': cgeru,
    'cupy::cscal': cscal,
    'cupy::csyrk': csyrk,
    'cupy::dasum': dasum,
    'cupy::daxpy': daxpy,
    'cupy::ddgmm': ddgmm,
    'cupy::ddot': ddot,
    'cupy::dgeam': dgeam,
    'cupy::dgemm': dgemm,
    'cupy::dgemv': dgemv,
    'cupy::dger': dger,
    'cupy::dnrm2': dnrm2,
    'cupy::dsbmv': dsbmv,
    'cupy::dscal': dscal,
    'cupy::dsyrk': dsyrk,
    'cupy::hgemm': hgemm,
    'cupy::sasum': sasum,
    'cupy::saxpy': saxpy,
    'cupy::sdgmm': sdgmm,
    'cupy::sdot': sdot,
    'cupy::sgeam': sgeam,
    'cupy::sgemm': sgemm,
    'cupy::sgemv': sgemv,
    'cupy::sger': sger,
    'cupy::snrm2': snrm2,
    'cupy::ssbmv': ssbmv,
    'cupy::sscal': sscal,
    'cupy::ssyrk': ssyrk,
    'cupy::zaxpy': zaxpy,
    'cupy::zdgmm': zdgmm,
    'cupy::zdotc': zdotc,
    'cupy::zdotu': zdotu,
    'cupy::zgeam': zgeam,
    'cupy::zgemm': zgemm,
    'cupy::zgemv': zgemv,
    'cupy::zgerc': zgerc,
    'cupy::zgeru': zgeru,
    'cupy::zscal': zscal,
    'cupy::zsyrk': zsyrk,
}

# PYTORCH_OPERATORS = BENCHMARK_OPERATORS
# PYTORCH_OPERATORS = V2_OPERATORS

# ============================================================
# VLLM OPERATORS (50 ops) - 算子名列表，函数延迟加载
# ============================================================
VLLM_OPERATOR_NAMES = [
    'allspark_w8a16_gemm',
    'apply_repetition_penalties_cuda',
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

# ============================================================
# CUBLAS OPERATORS (50 ops) - 算子名列表，函数延迟加载
# ============================================================
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

# ============================================================
# TORCH OPERATORS (110 ops) - 完整 V2_1 算子列表
# ============================================================
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

# ============================================================
# KernelGenBench - 算子名列表 (50 vllm13 + 50 cublas + 110 torch = 210)
# ============================================================
KERNELGENBENCH_OPERATOR_NAMES = (
    [f'vllm13::{name}' for name in VLLM_OPERATOR_NAMES] +
    [f'cublas::{name}' for name in CUBLAS_OPERATOR_NAMES] +
    [f'aten::{name}' for name in TORCH_OPERATOR_NAMES]
)


def _load_vllm_operators():
    """延迟加载 VLLM baseline 函数"""
    from .baseline import vllm13
    return {f'vllm13::{name}': getattr(vllm13, name) for name in VLLM_OPERATOR_NAMES}


def _load_cublas_operators():
    """延迟加载 CUBLAS baseline 函数"""
    from .baseline import cublas
    return {f'cublas::{name}': getattr(cublas, name) for name in CUBLAS_OPERATOR_NAMES}


def _load_torch_operators():
    """加载 torch aten 算子"""
    ops = {}
    for name in TORCH_OPERATOR_NAMES:
        op = getattr(torch.ops.aten, name, None)
        if op is not None:
            ops[f'aten::{name}'] = op
    return ops


def get_vllm_operators():
    """获取 VLLM_OPERATORS 字典"""
    return _load_vllm_operators()


def get_cublas_operators():
    """获取 CUBLAS_OPERATORS 字典"""
    return _load_cublas_operators()


def get_kernelgenbench_operators():
    """获取 KernelGenBench 算子字典 (50 vllm + 50 cublas + 110 torch = 210)"""
    ops = {}
    ops.update(_load_vllm_operators())
    ops.update(_load_cublas_operators())
    ops.update(_load_torch_operators())
    return ops


# ============ MM Shape-Specific Benchmark ============
# 每个 (shape, dtype) = 1 道独立题目，底层都是 aten::mm
# 50 shapes × 2 dtypes (f32, f16) = 100 题
# Shape 来源：2907 个 HF 模型真实训练 trace (aten.mm.default)

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
    """获取 MM Shape Benchmark 算子字典

    返回格式: {"aten::mm_128x768_768x768_f32": torch.ops.aten.mm, ...}
    所有 shape 题共用同一个底层算子 torch.ops.aten.mm。
    """
    mm_op = getattr(torch.ops.aten, "mm", None)
    if mm_op is None:
        raise RuntimeError("torch.ops.aten.mm not found")
    ops = {}
    for name in MM_SHAPE_OPERATOR_NAMES:
        ops[f"aten::{name}"] = mm_op
    return ops


def get_mm_shape_info(label: str):
    """获取 shape 信息: 返回 ((M, K), (K, N), dtype_short) 或 None"""
    return MM_SHAPE_OPERATORS.get(label)


# if os.environ.get("FLAGBENCH_USE_DYNAMIC_IMPL_INFO", "0") == "1":
#     dynamic_impl_info = DynamicImplInfo()
#     IMPL_INFO = dynamic_impl_info
dynamic_impl_info = DynamicImplInfo()
IMPL_INFO = dynamic_impl_info
    
if __name__ == "__main__":
    op_name_list = list(PYTORCH_OPERATORS.keys())
    dynamic_impl_info = DynamicImplInfo()
    namespace = "aten"
    for op_name in op_name_list:
        impl_info = dynamic_impl_info.get(namespace, op_name.split(".")[-1])
        print(f"{op_name}: {impl_info}")