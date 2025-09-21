import torch
import torch.nn as nn
from torch import Tensor

def make_linear(in_dim, out_dim):
    linear = nn.Linear(in_dim, out_dim, bias=False)
    nn.init.xavier_uniform_(linear.weight)
    return linear