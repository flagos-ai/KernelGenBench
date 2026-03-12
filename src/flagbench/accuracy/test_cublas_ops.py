import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("caxpy")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.complex64])
def test_caxpy_cublas_baseline(n, incx, incy, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    
    # Convert alpha to complex
    alpha = torch.tensor(alpha, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.caxpy(n, alpha, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.caxpy(n, alpha, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cdgmm")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.complex64])
def test_cdgmm_cublas_baseline(m, n, mode, dtype):
    # Initialize input tensors
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    x = torch.randn(n if mode == "right" else m, dtype=dtype, device='cuda')
    C = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1
    
    # Call baseline
    ref_out = flagbench.baseline.cdgmm(
        mode=mode_flag,
        m=m,
        n=n,
        A=A,
        lda=m,
        x=x,
        incx=1,
        C=C,
        ldc=m
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cdgmm(
        mode=mode_flag,
        m=m,
        n=n,
        A=A,
        lda=m,
        x=x,
        incx=1,
        C=C,
        ldc=m
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cdotc")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.complex64])
def test_cdotc_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cdotc(n, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.cdotc(n, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cdotu")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.complex64])
def test_cdotu_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cdotu(n, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.cdotu(n, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cgeam")
@parametrize("transa, transb", [('N', 'N'), ('T', 'N'), ('N', 'T')])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+2.0j)])
@parametrize("dtype", [torch.complex64])
def test_cgeam_cublas_baseline(transa, transb, M, N, alpha, beta, dtype):
    # Determine input shapes based on transpose flags
    A_shape = (M, N) if transa == 'N' else (N, M)
    B_shape = (M, N) if transb == 'N' else (N, M)
    
    # Initialize input tensors
    A = torch.randn(*A_shape, dtype=dtype, device='cuda')
    B = torch.randn(*B_shape, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgeam(
        transa, transb, M, N, alpha, A, A.shape[1], beta, B, B.shape[1], C, C.shape[1]
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgeam(
        transa, transb, M, N, alpha, A, A.shape[1], beta, B, B.shape[1], C, C.shape[1]
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 128, 128), (256, 512, 1024)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+0.0j)])
@parametrize("dtype", [torch.complex64])
def test_cgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.cgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cgemv")
@parametrize("trans", ["N", "T"])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+0.0j)])
@parametrize("dtype", [torch.complex64])
def test_cgemv_cublas_baseline(trans, M, N, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if trans == "N" else M, dtype=dtype, device='cuda')
    y = torch.randn(M if trans == "N" else N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgemv(
        trans,
        M,
        N,
        alpha,
        A,
        M,
        x,
        1,
        beta,
        y,
        1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgemv(
        trans,
        M,
        N,
        alpha,
        A,
        M,
        x,
        1,
        beta,
        y,
        1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cgerc")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex64])
def test_cgerc_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Convert alpha to complex
    alpha_complex = torch.tensor(complex(alpha, 0), dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.cgerc(
        m, n, alpha_complex, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgerc(
        m, n, alpha_complex, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cgeru")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex64])
def test_cgeru_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    lda = m
    
    # Call baseline
    ref_out = flagbench.baseline.cgeru(
        m, n, alpha, x, incx, y, incy, A, lda
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.cgeru(
        m, n, alpha, x, incx, y, incy, A, lda
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("cscal")
@parametrize("n", [16, 128, 1024])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
def test_cscal_cublas_baseline(n, incx, alpha):
    dtype = torch.complex64
    alpha = torch.tensor(alpha, dtype=torch.float32, device='cuda').view(torch.complex64)
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for in-place operations
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.cscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.cscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, dtype)
@label("csyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
@parametrize("dtype", [torch.complex64])
def test_csyrk_cublas_baseline(n, k, alpha, beta, uplo, trans, dtype):
    # Initialize input tensors
    A = torch.randn(n, k, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Convert alpha and beta to complex
    alpha = complex(alpha, 0)
    beta = complex(beta, 0)
    
    # Call baseline
    ref_out = flagbench.baseline.csyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=n,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.csyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=n,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dasum")
@parametrize("n, incx", [(10, 1), (1024, 1), (4096, 2), (8192, 4)])
@parametrize("dtype", [torch.float64])
def test_dasum_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.dasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("daxpy")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float64])
def test_daxpy_cublas_baseline(n, incx, incy, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.daxpy(n, alpha, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.daxpy(n, alpha, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("ddgmm")
@parametrize("mode", ["left", "right"])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float64])
def test_ddgmm_cublas_baseline(mode, M, N, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(M if mode == "left" else N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline mode
    ref_out = flagbench.baseline.ddgmm(
        mode=mode,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.ddgmm(
        mode=mode,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("ddot")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.float64])
def test_ddot_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.ddot(n, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.ddot(n, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dgeam")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
@parametrize("transa, transb", [("N", "N"), ("T", "N"), ("N", "T")])
def test_dgeam_cublas_baseline(M, N, alpha, beta, dtype, transa, transb):
    # Initialize input tensors
    lda = M if transa == "N" else N
    ldb = M if transb == "N" else N
    ldc = M
    
    A = torch.randn(M if transa == "N" else N, N if transa == "N" else M, dtype=dtype, device='cuda')
    B = torch.randn(M if transb == "N" else N, N if transb == "N" else M, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.dgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 256, 64), (512, 512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
def test_dgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.dgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dgemv")
@parametrize("trans", ["N", "T"])
@parametrize("M, N", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
def test_dgemv_cublas_baseline(trans, M, N, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if trans == "N" else M, dtype=dtype, device='cuda')
    y = torch.randn(M if trans == "N" else N, dtype=dtype, device='cuda')
    
    # Convert trans to cublasOperation_t
    trans_flag = 0 if trans == "N" else 1  # 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
    
    # Call baseline
    ref_out = flagbench.baseline.dgemv(
        trans_flag, M, N, alpha, A, M, x, 1, beta, y, 1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.dgemv(
        trans_flag, M, N, alpha, A, M, x, 1, beta, y, 1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dger")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.float64])
def test_dger_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    lda = max(1, m)
    
    # Make copies for baseline and triton calls
    A_baseline = A.clone()
    A_triton = A.clone()
    
    # Call baseline
    flagbench.baseline.dger(
        m, n, alpha, x, incx, y, incy, A_baseline, lda
    )
    
    # Call Triton implementation
    flagbench.triton.dger(
        m, n, alpha, x, incx, y, incy, A_triton, lda
    )
    
    # Compare results
    assert_close(A_triton, A_baseline, dtype)
@label("dnrm2")
@parametrize("n", [16, 128, 1024, 4096])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float64])
def test_dnrm2_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dnrm2(n, x.data_ptr(), incx, torch.zeros(1, dtype=dtype, device='cuda').data_ptr())
    
    # Call Triton implementation
    act_out = flagbench.triton.dnrm2(n, x.data_ptr(), incx, torch.zeros(1, dtype=dtype, device='cuda').data_ptr())
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dsbmv")
@parametrize("n, k", [(16, 2), (128, 16), (512, 32)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
@parametrize("uplo", ["L", "U"])
def test_dsbmv_cublas_baseline(n, k, alpha, beta, dtype, uplo):
    # Initialize input tensors
    lda = k + 1
    A = torch.randn((lda, n), dtype=dtype, device='cuda')
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    # Set incx and incy to 1 for simplicity
    incx = 1
    incy = 1
    
    # Call baseline
    ref_out = flagbench.baseline.dsbmv(
        uplo, n, k, alpha, A, lda, x, incx, beta, y, incy
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.dsbmv(
        uplo, n, k, alpha, A, lda, x, incx, beta, y, incy
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dscal")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float64])
def test_dscal_cublas_baseline(n, incx, alpha, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    ref_out = flagbench.baseline.dscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    act_out = flagbench.triton.dscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("dsyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float64])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
def test_dsyrk_cublas_baseline(n, k, alpha, beta, dtype, uplo, trans):
    # Initialize input tensors
    lda = n if trans == "N" else k
    A = torch.randn(n if trans == "N" else k, k if trans == "N" else n, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Call Triton implementation
    act_out = flagbench.triton.dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("hgemm")
@parametrize("M, N, K", [(16, 16, 16), (128, 128, 64), (256, 256, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float16])
def test_hgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.hgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.hgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sasum")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float32])
def test_sasum_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.sasum(n, x, incx, torch.zeros(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("saxpy")
@parametrize("n", [16, 1024, 65536])
@parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
@parametrize("incx, incy", [(1, 1), (1, 2), (2, 1)])
@parametrize("dtype", [torch.float32])
def test_saxpy_cublas_baseline(n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for in-place operation
    y_ref = y.clone()
    y_act = y.clone()
    
    # Call baseline
    flagbench.baseline.saxpy(n, alpha, x, incx, y_ref, incy)
    
    # Call Triton implementation
    flagbench.triton.saxpy(n, alpha, x, incx, y_act, incy)
    
    # Compare results
    assert_close(y_act, y_ref, dtype)
@label("sdgmm")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.float32])
def test_sdgmm_cublas_baseline(M, N, mode, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(M if mode == "left" else N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1
    
    # Call baseline
    ref_out = flagbench.baseline.sdgmm(
        mode_flag, M, N, A, M, x, 1, C, M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sdgmm(
        mode_flag, M, N, A, M, x, 1, C, M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sdot")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.float32])
def test_sdot_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    result = torch.empty(1, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sdot(n, x, incx, y, incy, result)
    
    # Call Triton implementation
    act_out = flagbench.triton.sdot(n, x, incx, y, incy, result)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sgeam")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
@parametrize("transa, transb", [("N", "N"), ("T", "N")])
def test_sgeam_cublas_baseline(M, N, alpha, beta, dtype, transa, transb):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    B = torch.randn(M, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sgeam(
        transa, transb, M, N, alpha, A, M, beta, B, N, C, M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgeam(
        transa, transb, M, N, alpha, A, M, beta, B, N, C, M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 256, 64), (512, 512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("transa, transb", [("N", "N"), ("T", "N"), ("N", "T")])
@parametrize("dtype", [torch.float32])
def test_sgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, dtype):
    # Initialize input tensors based on transpose flags
    A_shape = (K, M) if transa == "T" else (M, K)
    B_shape = (N, K) if transb == "T" else (K, N)
    
    A = torch.randn(*A_shape, dtype=dtype, device='cuda')
    B = torch.randn(*B_shape, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.sgemm(
        transa, transb, M, N, K, alpha, A, A_shape[1], B, B_shape[1], beta, C, N
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgemm(
        transa, transb, M, N, K, alpha, A, A_shape[1], B, B_shape[1], beta, C, N
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sgemv")
@parametrize("trans", ["N", "T"])
@parametrize("m, n", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
def test_sgemv_cublas_baseline(trans, m, n, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    x = torch.randn(n if trans == "N" else m, dtype=dtype, device='cuda')
    y = torch.randn(m if trans == "N" else n, dtype=dtype, device='cuda')
    
    # Convert trans to cublasOperation_t
    trans_flag = 0 if trans == "N" else 1  # 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
    
    # Call baseline
    ref_out = flagbench.baseline.sgemv(
        trans_flag,
        m,
        n,
        alpha,
        A.data_ptr(),
        m,
        x.data_ptr(),
        1,
        beta,
        y.data_ptr(),
        1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.sgemv(
        trans_flag,
        m,
        n,
        alpha,
        A.data_ptr(),
        m,
        x.data_ptr(),
        1,
        beta,
        y.data_ptr(),
        1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sger")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float32])
def test_sger_cublas_baseline(m, n, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Make copies for in-place operation
    ref_A = A.clone()
    act_A = A.clone()
    
    # Call baseline
    flagbench.baseline.sger(
        m, n, alpha,
        x, 1,  # incx=1
        y, 1,  # incy=1
        ref_A, m  # lda=m (leading dimension)
    )
    
    # Call Triton implementation
    flagbench.triton.sger(
        m, n, alpha,
        x, 1,  # incx=1
        y, 1,  # incy=1
        act_A, m  # lda=m
    )
    
    # Compare results
    assert_close(act_A, ref_A, dtype)
@label("snrm2")
@parametrize("n", [16, 128, 1024, 4096])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float32])
def test_snrm2_cublas_baseline(n, incx, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.snrm2(n, x, incx, torch.empty(1, dtype=dtype, device='cuda'))
    
    # Call Triton implementation
    act_out = flagbench.triton.snrm2(n, x, incx, torch.empty(1, dtype=dtype, device='cuda'))
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("ssbmv")
@parametrize("n, k", [(16, 2), (128, 16), (512, 32)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
@parametrize("uplo", ["L", "U"])
def test_ssbmv_cublas_baseline(n, k, alpha, beta, dtype, uplo):
    # Initialize input tensors
    lda = k + 1
    A = torch.randn((lda, n), dtype=dtype, device='cuda')
    x = torch.randn(n, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    
    # Convert uplo to cublasFillMode_t
    uplo_flag = 0 if uplo == "L" else 1  # 0 for lower, 1 for upper
    
    # Call baseline
    ref_out = flagbench.baseline.ssbmv(
        uplo_flag, n, k, alpha, A, lda, x, 1, beta, y, 1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.ssbmv(
        uplo_flag, n, k, alpha, A, lda, x, 1, beta, y, 1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("sscal")
@parametrize("n", [16, 1024, 4096])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("dtype", [torch.float32])
def test_sscal_cublas_baseline(n, incx, alpha, dtype):
    # Initialize input tensor
    x = torch.randn(n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton since operation is in-place
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.sscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.sscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, dtype)
@label("ssyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.float32])
def test_ssyrk_cublas_baseline(n, k, uplo, trans, alpha, beta, dtype):
    # Initialize input tensors
    lda = k if trans == "N" else n
    A = torch.randn(n, k, dtype=dtype, device='cuda') if trans == "N" else torch.randn(k, n, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Call Triton implementation
    act_out = flagbench.triton.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C, n)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zaxpy")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("alpha", [1.0, 0.5, 1.5+2.0j])
@parametrize("dtype", [torch.complex128])
def test_zaxpy_cublas_baseline(n, incx, incy, alpha, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton since operation is in-place
    y_ref = y.clone()
    y_act = y.clone()
    
    # Call baseline
    flagbench.baseline.zaxpy(n, alpha, x, incx, y_ref, incy)
    
    # Call Triton implementation
    flagbench.triton.zaxpy(n, alpha, x, incx, y_act, incy)
    
    # Compare results
    assert_close(y_act, y_ref, dtype)
@label("zdgmm")
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("mode", ["left", "right"])
@parametrize("dtype", [torch.complex128])
def test_zdgmm_cublas_baseline(M, N, mode, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if mode == "right" else M, dtype=dtype, device='cuda')
    C = torch.empty_like(A)
    
    # Convert mode to cublasSideMode_t equivalent
    mode_flag = 0 if mode == "left" else 1  # 0=CUBLAS_SIDE_LEFT, 1=CUBLAS_SIDE_RIGHT
    
    # Call baseline
    ref_out = flagbench.baseline.zdgmm(
        mode=mode_flag,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zdgmm(
        mode=mode_flag,
        m=M,
        n=N,
        A=A,
        lda=M,
        x=x,
        incx=1,
        C=C,
        ldc=M
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zdotc")
@parametrize("n", [2, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.complex128])
def test_zdotc_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zdotc(n, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.zdotc(n, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zdotu")
@parametrize("n", [16, 128, 1024])
@parametrize("incx, incy", [(1, 1), (2, 1), (1, 2)])
@parametrize("dtype", [torch.complex128])
def test_zdotu_cublas_baseline(n, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(n * incx, dtype=dtype, device='cuda')
    y = torch.randn(n * incy, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zdotu(n, x, incx, y, incy)
    
    # Call Triton implementation
    act_out = flagbench.triton.zdotu(n, x, incx, y, incy)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zgeam")
@parametrize("transa, transb", [(0, 0), (1, 0), (0, 1), (1, 1)])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.complex128])
def test_zgeam_cublas_baseline(transa, transb, M, N, alpha, beta, dtype):
    # Initialize input tensors
    lda = M if transa == 0 else N
    ldb = M if transb == 0 else N
    ldc = M
    
    A = torch.randn((M, N) if transa == 0 else (N, M), dtype=dtype, device='cuda')
    B = torch.randn((M, N) if transb == 0 else (N, M), dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zgeam(
        transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zgemm")
@parametrize("M, N, K", [(2, 3, 4), (128, 256, 64), (512, 512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.complex128])
def test_zgemm_cublas_baseline(M, N, K, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    C = torch.randn(M, N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Call Triton implementation
    act_out = flagbench.triton.zgemm(A, B, C, alpha=alpha, beta=beta)
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zgemv")
@parametrize("trans", ["N", "T"])
@parametrize("M, N", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0+0.0j, 0.0+0.0j), (0.5+0.5j, 0.5-0.5j), (2.0+1.0j, 1.0+2.0j)])
@parametrize("dtype", [torch.complex128])
def test_zgemv_cublas_baseline(trans, M, N, alpha, beta, dtype):
    # Initialize input tensors
    A = torch.randn(M, N, dtype=dtype, device='cuda')
    x = torch.randn(N if trans == "N" else M, dtype=dtype, device='cuda')
    y = torch.randn(M if trans == "N" else N, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgemv(
        trans,
        M,
        N,
        alpha,
        A,
        M,
        x,
        1,
        beta,
        y,
        1
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zgemv(
        trans,
        M,
        N,
        alpha,
        A,
        M,
        x,
        1,
        beta,
        y,
        1
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zgerc")
@parametrize("m, n", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha", [1.0, 0.5, 2.0])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex128])
def test_zgerc_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Make copies for baseline and triton calls
    A_ref = A.clone()
    A_act = A.clone()
    
    # Call baseline
    flagbench.baseline.zgerc(
        m, n, 
        alpha, 
        x, incx, 
        y, incy, 
        A_ref, m
    )
    
    # Call Triton implementation
    flagbench.triton.zgerc(
        m, n, 
        alpha, 
        x, incx, 
        y, incy, 
        A_act, m
    )
    
    # Compare results
    assert_close(A_act, A_ref, dtype)
@label("zgeru")
@parametrize("m, n", [(2, 3), (128, 256), (1024, 512)])
@parametrize("alpha", [1.0, 0.5, 0.0+1.0j])
@parametrize("incx, incy", [(1, 1), (2, 3)])
@parametrize("dtype", [torch.complex128])
def test_zgeru_cublas_baseline(m, n, alpha, incx, incy, dtype):
    # Initialize input tensors
    x = torch.randn(m, dtype=dtype, device='cuda')
    y = torch.randn(n, dtype=dtype, device='cuda')
    A = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zgeru(
        m, n, alpha, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zgeru(
        m, n, alpha, 
        x, incx, 
        y, incy, 
        A, max(1, m)
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
@label("zscal")
@parametrize("n", [2, 128, 1024])
@parametrize("incx", [1, 2, 3])
@parametrize("alpha", [1.0, 0.5, 1.5+2.0j])
def test_zscal_cublas_baseline(n, incx, alpha):
    # Initialize input tensor
    x = torch.randn(n * incx, dtype=torch.complex128, device='cuda')
    alpha = torch.tensor(alpha, dtype=torch.complex128, device='cuda')
    
    # Make copies for in-place operations
    x_ref = x.clone()
    x_act = x.clone()
    
    # Call baseline
    flagbench.baseline.zscal(n, alpha, x_ref, incx)
    
    # Call Triton implementation
    flagbench.triton.zscal(n, alpha, x_act, incx)
    
    # Compare results
    assert_close(x_act, x_ref, torch.complex128)
@label("zsyrk")
@parametrize("n, k", [(2, 3), (128, 256), (512, 512)])
@parametrize("alpha, beta", [(1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
@parametrize("dtype", [torch.complex128])
@parametrize("uplo, trans", [("L", "N"), ("U", "T")])
def test_zsyrk_cublas_baseline(n, k, alpha, beta, dtype, uplo, trans):
    # Initialize input tensors
    lda = n if trans == "N" else k
    A = torch.randn(n, k, dtype=dtype, device='cuda')
    C = torch.randn(n, n, dtype=dtype, device='cuda')
    
    # Call baseline
    ref_out = flagbench.baseline.zsyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=lda,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Call Triton implementation
    act_out = flagbench.triton.zsyrk(
        uplo=uplo,
        trans=trans,
        n=n,
        k=k,
        alpha=alpha,
        A=A,
        lda=lda,
        beta=beta,
        C=C,
        ldc=n
    )
    
    # Compare results
    assert_close(act_out, ref_out, dtype)
