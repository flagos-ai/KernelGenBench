#!/bin/bash
# ============================================================================
# FlagBench - Kaldi CUDA Bindings Build Script
# ============================================================================
# 
# 功能：自动化构建Kaldi CUDA算子的PyTorch bindings
# 
# 阶段：
#   1. extract_cuda_kernels.py   - 从CUDA头文件提取算子定义
#   2. generate_binding_code.py  - 生成C++ binding代码
#   3. compile_bindings.py       - 编译成.so库
#   4. test_bindings.py          - 测试bindings
#
# 使用方法：
#   bash script/build_all.sh [--skip-compile] [--skip-test]
#
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 分隔线
print_separator() {
    echo "============================================================================"
}

# 检查Python环境
check_python_env() {
    log_info "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python not found!"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_success "Python version: $PYTHON_VERSION"
    
    # 检查PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        log_error "PyTorch not found! Please install PyTorch first."
        exit 1
    fi
    
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    log_success "PyTorch version: $TORCH_VERSION"
}

# 解析命令行参数
SKIP_COMPILE=0
SKIP_TEST=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-compile)
            SKIP_COMPILE=1
            shift
            ;;
        --skip-test)
            SKIP_TEST=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-compile    Skip compilation stage (stage 3)"
            echo "  --skip-test       Skip testing stage (stage 4)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

print_separator
echo -e "${GREEN}FlagBench - Kaldi CUDA Bindings Builder${NC}"
echo "Project root: $PROJECT_ROOT"
print_separator

# 检查环境
check_python_env

# ============================================================================
# 阶段1: 提取CUDA算子定义
# ============================================================================
print_separator
log_info "Stage 1: Extracting CUDA kernel definitions..."
print_separator

python script/extract_cuda_kernels.py \
    --input kaldi/src/cudamatrix/cu-kernels-ansi.h \
    --output csrc/kaldi_kernels.json \
    --cu-impl kaldi/src/cudamatrix/cu-kernels.cu

if [ $? -eq 0 ]; then
    log_success "Stage 1 completed successfully"
else
    log_error "Stage 1 failed"
    exit 1
fi

# ============================================================================
# 阶段2: 生成C++ binding代码
# ============================================================================
print_separator
log_info "Stage 2: Generating C++ binding code..."
print_separator

python script/generate_binding_code.py \
    --input csrc/kaldi_kernels.json \
    --output csrc/kaldi_ops.cpp \
    --cuda-src kaldi/src/cudamatrix \
    --namespace kaldi \
    --setup-output csrc/setup.py

if [ $? -eq 0 ]; then
    log_success "Stage 2 completed successfully"
else
    log_error "Stage 2 failed"
    exit 1
fi

# ============================================================================
# 阶段3: 编译bindings
# ============================================================================
if [ $SKIP_COMPILE -eq 0 ]; then
    print_separator
    log_info "Stage 3: Compiling bindings..."
    print_separator
    
    python script/compile_bindings.py \
        --csrc-dir csrc \
        --kaldi-src kaldi/src \
        --output-dir lib
    
    if [ $? -eq 0 ]; then
        log_success "Stage 3 completed successfully"
    else
        log_warning "Stage 3 failed or skipped (compilation may need manual intervention)"
        log_warning "You may need to:"
        log_warning "  1. Compile Kaldi library first"
        log_warning "  2. Adjust include/library paths"
        log_warning "  3. Check CUDA installation"
        # 不退出，继续后续步骤
    fi
else
    log_info "Stage 3 skipped (--skip-compile)"
fi

# ============================================================================
# 阶段4: 测试bindings
# ============================================================================
if [ $SKIP_TEST -eq 0 ]; then
    print_separator
    log_info "Stage 4: Testing bindings..."
    print_separator
    
    # 检查.so文件是否存在
    SO_FILE=$(find lib -name "*.so" 2>/dev/null | head -n 1)
    
    if [ -z "$SO_FILE" ]; then
        log_warning "No .so file found in lib/, skipping tests"
        log_warning "Please compile the bindings first (stage 3)"
    else
        log_info "Found library: $SO_FILE"
        
        python script/test_bindings.py \
            --lib "$SO_FILE" \
            --namespace kaldi \
            --report test_report.json
        
        if [ $? -eq 0 ]; then
            log_success "Stage 4 completed successfully"
        else
            log_warning "Stage 4 failed (some tests may have failed)"
        fi
    fi
else
    log_info "Stage 4 skipped (--skip-test)"
fi

# ============================================================================
# 总结
# ============================================================================
print_separator
log_success "Build process completed!"
print_separator

echo ""
echo "Generated files:"
echo "  - csrc/kaldi_kernels.json     (Stage 1: Kernel definitions)"
echo "  - csrc/kaldi_ops.cpp          (Stage 2: C++ binding code)"
echo "  - csrc/setup.py               (Stage 2: Build configuration)"
echo "  - lib/*.so                    (Stage 3: Compiled library)"
echo "  - test_report.json            (Stage 4: Test results)"
echo ""

if [ $SKIP_COMPILE -eq 0 ]; then
    echo "Next steps:"
    echo "  1. Check test_report.json for test results"
    echo "  2. Load the library in Python:"
    echo "       import torch"
    echo "       torch.ops.load_library('lib/kaldi_ops.so')"
    echo "       torch.ops.kaldi.add_row_sum_mat(...)  # Use your operators"
    echo ""
else
    echo "Compilation was skipped. To compile, run:"
    echo "  bash script/build_all.sh"
    echo ""
fi

log_success "All done! 🎉"
