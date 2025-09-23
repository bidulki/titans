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
def gelu_bakward(x: torch.Tensor) -> torch.Tensor:
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff


# SiLU (Swish) backward: d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
def silu_backward(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))
