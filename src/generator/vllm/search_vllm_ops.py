# ============================================================
# 1. torch.ops.vllm / _vllm
# ============================================================
def dump_torch_ops():
    result = {}
    for ns_name in ["vllm", "_vllm"]:
        ns = getattr(torch.ops, ns_name, None)
        if ns is None:
            continue
        ops = []
        for name in dir(ns):
            if name.startswith("_"):
                continue
            try:
                op = getattr(ns, name)
                schema = None
                if hasattr(op, "default"):
                    try:
                        schema = str(op.default.schema)
                    except Exception:
                        pass
                ops.append({"name": name, "schema": schema})
            except Exception as e:
                ops.append({"name": name, "error": str(e)})
        result[ns_name] = ops
    return result
# ============================================================
# 2. vllm/csrc/ops.h (native CUDA op interface)
# ============================================================
def dump_csrc_ops_h(vllm_pkg):
    root = os.path.abspath("/root/Git/vllm")
    ops_h = os.path.join(root, "csrc", "ops.h")
    if not os.path.exists(ops_h):
        return []
    with open(ops_h, "r") as f:
        text = f.read()
    pattern = re.compile(
        r"""
        ^\s*
        (?:void|torch::Tensor|std::vector<torch::Tensor>)
        \s+
        ([a-zA-Z_][a-zA-Z0-9_]*)
        \s*\(
        ([^;]*?)
        \)
        \s*;
        """,
        re.MULTILINE | re.VERBOSE,
    )
    ops = []
    for m in pattern.finditer(text):
        ops.append({
            "name": m.group(1),
            "args": " ".join(m.group(2).split()),
            "source": "csrc/ops.h",
        })
    return ops
# ============================================================
# 3. Triton kernel definitions (SAFE static scan)
# ============================================================
def is_triton_kernel_definition(obj):
    if not inspect.isfunction(obj):
        return False
    try:
        return "triton" in obj.__globals__
    except Exception:
        return False
def dump_triton_definitions(vllm_pkg):
    kernels = []
    root = os.path.dirname(vllm_pkg.__file__)
    for m in pkgutil.walk_packages([root], prefix="vllm."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for name, obj in vars(mod).items():
            if is_triton_kernel_definition(obj):
                try:
                    src = inspect.getsourcefile(obj)
                    sig = str(inspect.signature(obj))
                except Exception:
                    src, sig = None, None
                kernels.append({
                    "module": m.name,
                    "name": name,
                    "signature": sig,
                    "source_file": src,
                })
    return kernels