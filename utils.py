import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

def make_linear(in_dim, out_dim):
    linear = nn.Linear(in_dim, out_dim, bias=False)
    nn.init.xavier_uniform_(linear.weight)
    return linear

class ParameterVectorizer:
    """Flatten/unflatten parameters of an nn.Module (fixed order)."""
    def __init__(self, module: nn.Module) -> None:
        names, shapes, numels = [], [], []
        for name, p in module.named_parameters():
            names.append(name); shapes.append(p.shape); numels.append(p.numel())
        self.names   = names
        self.shapes  = shapes
        self.numels  = numels
        self.total   = sum(numels)

    def flatten(self, params: OrderedDict[str, Tensor]) -> Tensor:
        return torch.cat([params[n].reshape(-1) for n in self.names], dim=0)

    def unflatten(self, vec: Tensor) -> OrderedDict[str, Tensor]:
        if vec.shape[-1] != self.total:
            raise ValueError("Bad vector size for unflatten")
        out: OrderedDict[str, Tensor] = OrderedDict()
        off = 0
        for n, shape, k in zip(self.names, self.shapes, self.numels):
            out[n] = vec.narrow(-1, off, k).view(*shape)
            off += k
        return out