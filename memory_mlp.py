from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gelu_bakward, make_parameter, silu_backward


@dataclass
class MemoryState:
    W: list[torch.Tensor]
    b: list[torch.Tensor]
    SW: list[torch.Tensor]
    Sb: list[torch.Tensor]


# MemoryMLP implementation
class MemoryMLP(nn.Module):
    """
    메모리를 담당하는 fast-weight MLP
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int = 2,
        activation: Literal["gelu", "silu"] = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.activation: Literal["gelu", "silu"] = activation
        self.depth = depth

        in_dim = dim
        self._W0 = nn.ParameterList()
        self._b0 = nn.ParameterList()
        for _ in range(depth - 1):
            self._W0.append(make_parameter(in_dim, hidden_dim))
            self._b0.append(make_parameter(in_dim, hidden_dim, init="zero"))
            in_dim = hidden_dim
        self._W0.append(make_parameter(in_dim, dim))
        self._b0.append(make_parameter(in_dim, dim, init="zero"))

    def init_state(self, batch_size: int, device=None) -> MemoryState:
        if device is None:
            device = self._W0[0].device

        W, b, SW, Sb = [], [], [], []
        for pW, pb in zip(self._W0, self._b0):
            Wi = pW.to(device).unsqueeze(0).expand(batch_size, -1, -1).clone()
            bi = pb.to(device).unsqueeze(0).expand(batch_size, -1, -1).clone()
            W.append(Wi)
            b.append(bi)
            SW.append(torch.zeros_like(Wi))
            Sb.append(torch.zeros_like(bi))
        return MemoryState(W=W, b=b, SW=SW, Sb=Sb)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh") if self.activation == "gelu" else F.silu(x)

    def _act_backward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_bakward(x) if self.activation == "gelu" else silu_backward(x)

    def step(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: MemoryState,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, MemoryState, torch.Tensor]:
        """
        Read + Update 수행
        """

        # cache for backward
        h_list: list[torch.Tensor] = []
        z_list: list[torch.Tensor] = []
        h = k
        h_list.append(h)

        # forward
        for i in range(len(state.W)):
            W, b = state.W[i], state.b[i]
            z = torch.einsum("bi,bij->bj", h, W) + b.squeeze(1)
            z_list.append(z)
            h = self._act(z) if i < self.depth - 1 else z
            h_list.append(h)

        # loss
        v_hat = h_list[-1]
        diff = v_hat - v
        loss = (diff * diff).mean()

        # backward
        delta = 2.0 * diff
        dW_list: list[torch.Tensor] = [None] * len(state.W)
        db_list: list[torch.Tensor] = [None] * len(state.b)

        h_prev = h_list[-2]
        dW_list[-1] = torch.einsum("bi,bj->bij", h_prev, delta)
        db_list[-1] = delta.unsqueeze(1)

        for i in reversed(range(len(state.W) - 1)):
            z = z_list[i]
            dh = torch.einsum("bi,bij->bj", delta, state.W[i + 1])
            delta = dh * self._act_backward(z)
            h_prev = h_list[i]
            dW_list[i] = torch.einsum("bi,bj->bij", h_prev, delta)
            db_list[i] = delta.unsqueeze(1)

        # update
        new_W, new_b, new_SW, new_Sb = [], [], [], []
        for i in range(len(state.W)):
            SW_i = eta * state.SW[i] - theta * dW_list[i]
            Sb_i = eta * state.Sb[i] - theta * db_list[i]
            W_i = (1.0 - alpha) * state.W[i] + SW_i
            b_i = (1.0 - alpha) * state.b[i] + Sb_i
            new_SW.append(SW_i)
            new_Sb.append(Sb_i)
            new_W.append(W_i)
            new_b.append(b_i)

        new_state = MemoryState(W=new_W, b=new_b, SW=new_SW, Sb=new_Sb)
        return v_hat, new_state, loss
