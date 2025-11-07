from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gelu_backward, make_parameter, silu_backward


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
        activation: Literal["gelu", "silu"] = "silu",
        use_bias: bool = True,
    ):
        super().__init__()
        assert depth >= 1

        self.activation: Literal["gelu", "silu"] = activation
        self.depth = depth
        self.use_bias = use_bias

        in_dim = dim
        self._W0 = nn.ParameterList()
        self._b0 = nn.ParameterList()

        for _ in range(depth - 1):
            self._W0.append(make_parameter(in_dim, hidden_dim))
            if self.use_bias:
                self._b0.append(make_parameter(1, hidden_dim, init="zero"))
            in_dim = hidden_dim

        self._W0.append(make_parameter(in_dim, dim))
        if self.use_bias:
            self._b0.append(make_parameter(1, dim, init="zero"))

    def init_state(self, batch_size: int, device=None) -> MemoryState:
        if device is None:
            device = self._W0[0].device

        W, b, SW, Sb = [], [], [], []

        for idx, pW in enumerate(self._W0):
            Wi = (
                pW.to(device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            )  # (B, in, out)
            W.append(Wi)
            SW.append(torch.zeros_like(Wi))

            if self.use_bias:
                pb = self._b0[idx].to(device)
                bi = (
                    pb.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                )  # (B, 1, out)
                b.append(bi)
                Sb.append(torch.zeros_like(bi))

        return MemoryState(W=W, b=b, SW=SW, Sb=Sb)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh") if self.activation == "gelu" else F.silu(x)

    def _act_backward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_backward(x) if self.activation == "gelu" else silu_backward(x)

    def retrieve(self, q: torch.Tensor, state: MemoryState):
        h = q  # (B, c, d)
        for i in range(self.depth):
            W = state.W[i]  # (B, in, out)
            z = torch.einsum("bci,bij->bcj", h, W)  # (B, c, out)
            if self.use_bias:
                z = z + state.b[i]  # (B, c, out)
            h = self._act(z) if i < self.depth - 1 else z
        return h

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: MemoryState,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, MemoryState, torch.Tensor]:
        B, c, _ = k.shape

        h_list, z_list, v_hat = self._forward_chunk(k, state)

        dW_tokens, db_tokens, loss = self._backward_chunk(
            k=k, v=v, state=state, h_list=h_list, z_list=z_list, v_hat=v_hat
        )

        new_W: list[torch.Tensor] = []
        new_b: list[torch.Tensor] = []
        new_SW: list[torch.Tensor] = []
        new_Sb: list[torch.Tensor] = []

        for i in range(self.depth):
            W_i = state.W[i]
            SW_i = state.SW[i]

            if self.use_bias:
                b_i = state.b[i]
                Sb_i = state.Sb[i]

            for t in range(c):
                grad_W_t = dW_tokens[i][:, t, :, :]  # (B, in, out)

                eta_t = eta[:, t]
                alpha_t = alpha[:, t]
                theta_t = theta[:, t]

                SW_i = eta_t * SW_i - theta_t * grad_W_t
                W_i = (1.0 - alpha_t) * W_i + SW_i

                if self.use_bias:
                    grad_b_t = db_tokens[i][:, t, :, :]
                    Sb_i = eta_t * Sb_i - theta_t * grad_b_t  # type: ignore
                    b_i = (1.0 - alpha_t) * b_i + Sb_i  # type: ignore

            new_W.append(W_i)
            new_SW.append(SW_i)

            if self.use_bias:
                new_b.append(b_i)  # type: ignore
                new_Sb.append(Sb_i)  # type: ignore

        new_state = MemoryState(W=new_W, b=new_b, SW=new_SW, Sb=new_Sb)

        return v_hat, new_state, loss

    def _forward_chunk(
        self, k: torch.Tensor, state: MemoryState
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        h_list: list[torch.Tensor] = []
        z_list: list[torch.Tensor] = []

        h = k  # (B, c, d)
        h_list.append(h)

        for i in range(self.depth):
            W = state.W[i]  # (B, in, out)
            z = torch.einsum("bci,bij->bcj", h, W)  # (B, c, out)
            if self.use_bias:
                z = z + state.b[i]
            z_list.append(z)

            h = self._act(z) if i < self.depth - 1 else z
            h_list.append(h)

        v_hat = h_list[-1]
        return h_list, z_list, v_hat

    def _backward_chunk(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        state: MemoryState,
        h_list: list[torch.Tensor],
        z_list: list[torch.Tensor],
        v_hat: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        diff = v_hat - v
        loss = (diff * diff).mean()

        delta = 2.0 * diff

        B, c, _ = k.shape

        delta = 2.0 * diff  # dL/d(v_hat)
        dW_tokens: list[torch.Tensor] = [torch.empty(0, device=k.device)] * self.depth
        db_tokens: list[torch.Tensor] = (
            [torch.empty(0, device=k.device)] * self.depth if self.use_bias else []
        )

        last = self.depth - 1
        h_prev = h_list[last]
        dW_tokens[last] = torch.einsum(
            "bci,bcj->bcij", h_prev, delta
        )  # (B, c, in, out)
        if self.use_bias:
            db_tokens[last] = delta.unsqueeze(2)  # (B, c, 1, out)

        if self.depth > 1:
            W_last = state.W[last]  # (B, in, out)
            delta = torch.einsum("bcj,bij->bcj", delta, W_last)  # (B, c, in)

        for i in reversed(range(self.depth - 1)):
            z = z_list[i]
            delta = delta * self._act_backward(z)  # (B, c, out)

            h_prev = h_list[i]
            dW_tokens[i] = torch.einsum("bci,bcj->bcij", h_prev, delta)

            if self.use_bias:
                db_tokens[i] = delta.unsqueeze(2)

            if i > 0:
                W_i = state.W[i]
                delta = torch.einsum("bcj,bij->bci", delta, W_i)

        return dW_tokens, db_tokens, loss
