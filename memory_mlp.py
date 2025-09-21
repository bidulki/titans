import torch.nn as nn
from utils import make_linear

# MemoryMLP implementation
class MemoryMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers = []
        in_dim = dim
        
        for i in range(depth - 1):
            layers += [make_linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
            
        layers += [make_linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
