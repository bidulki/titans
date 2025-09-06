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
        for i in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
