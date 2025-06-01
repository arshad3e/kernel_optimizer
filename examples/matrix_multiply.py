import torch

class Model:
    def __init__(self, N):
        self.N = N

    def forward(self, A, B):
        return torch.matmul(A, B)

def get_inputs(N):
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    return A, B

def get_init_inputs():
    return {"N": 1024}
