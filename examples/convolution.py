import torch
import torch.nn as nn

class Model:
    def __init__(self):
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

def get_inputs(N):
    x = torch.randn(1, 3, N, N, device='cuda', dtype=torch.float32)
    return x

def get_init_inputs():
    return {"N": 224}
