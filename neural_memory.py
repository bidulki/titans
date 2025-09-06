import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from memory_mlp import MemoryMLP

# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.dim = dim
        self.memory = MemoryMLP(dim, hidden, depth)

    def retrieve(self):
        pass

    def forward(self):
        pass

    def update(self):
        pass

