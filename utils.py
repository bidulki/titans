from typing import Literal

import torch
import torch.nn as nn


def make_linear(in_dim: int, out_dim: int, bias: bool = False) -> nn.Linear:
    linear = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(linear.weight)
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


def make_parameter(
    in_dim: int, out_dim: int, init: Literal["xavier", "zero"] = "xavier"
) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(in_dim, out_dim))
    match init:
        case "xavier":
            nn.init.xavier_uniform_(parameter)
        case "zero":
            nn.init.zeros_(parameter)
    return parameter


# from TTT pytorch
@torch.jit.script
def gelu_backward(x: torch.Tensor) -> torch.Tensor:
    c0 = 0.79788456  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    t = torch.tanh(c0 * (x + c1 * x3))
    dt = (1 - t * t) * (c0 * (1 + 3 * c1 * x * x))
    return 0.5 * (1 + t) + 0.5 * x * dt


@torch.jit.script
def silu_backward(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) backward: d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))
