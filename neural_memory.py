import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_mlp import MemoryMLP
from utils import make_linear

# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim=16, 
        memory_hidden_dim=32, 
        memory_depth=2
    ):
        super().__init__()
        self.dim = dim

        self.memory = MemoryMLP(dim, memory_hidden_dim, memory_depth)
        self.key_proj = make_linear(dim, dim)
        self.value_proj = make_linear(dim, dim)

    def retrieve(self):
        pass

    def forward(self):
        pass

    def update(self):
        pass

