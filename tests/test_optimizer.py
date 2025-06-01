import pytest
from src.kernel_optimizer import run_kernelbench

def test_run_kernelbench():
    sample_kernel = """
    __global__ void dummy_kernel(float *A, float *B, float *C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) C[idx] = A[idx] + B[idx];
    }
    """
    sample_pytorch = """
    import torch
    def forward(A, B):
        return A + B
    """
    result = run_kernelbench(sample_kernel, sample_pytorch, 1024)
    assert isinstance(result, dict)
    assert "fast_p" in result
    assert "correct" in result
