import torch 
import gpytorch
from gpytorch.kernels import Kernel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class PolynomialDimerKernel(Kernel):
    is_stationary = False

    def __init__(self, arg_num_dims=None, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        pass
