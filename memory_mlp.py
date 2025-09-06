import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# MemoryMLP implementation
class MemoryMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        in_dim = dim

        def make_linear(in_dim, out_dim):
            linear = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_uniform_(linear.weight)
            return linear
        
        for i in range(depth - 1):
            layers += [make_linear(in_dim, dim), nn.SiLU()]
            in_dim = dim
            
        layers += [make_linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
