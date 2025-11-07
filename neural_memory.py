import torch
import torch.nn as nn
import torch.nn.functional as F

from depthwise_separable_conv import CausalDSC1d
from memory_mlp import MemoryMLP, MemoryState
from utils import make_linear


# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        memory_hidden_dim: int,
        memory_depth: int = 2,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        self.memory = MemoryMLP(dim, memory_hidden_dim, memory_depth)

        self.Wk = make_linear(dim, dim)
        self.Wv = make_linear(dim, dim)
        self.Wq = make_linear(dim, dim)

        self.hyper = nn.Linear(dim, 3)

        self.conv_q = CausalDSC1d(dim, dim)
        self.conv_k = CausalDSC1d(dim, dim)
        self.conv_v = CausalDSC1d(dim, dim)

    def _l2norm(self, x: torch.Tensor, eps: float = 1e-6):
        return F.normalize(x, dim=-1, eps=eps)

    def _hyper(self, x: torch.Tensor):
        h = self.hyper(x)
        eta = torch.sigmoid(h[..., 0:1]).unsqueeze(-1)
        alpha = torch.sigmoid(h[..., 1:2]).unsqueeze(-1)
        theta = (F.softplus(h[..., 2:3]) * 1e-2 + 1e-6).unsqueeze(-1)
        return eta, alpha, theta

    def retrieve(
        self, x: torch.Tensor, state: MemoryState | None = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device
        if state is None:
            state = self.memory.init_state(B, device=device)

        q = self.Wq(x)
        q = self.conv_q(q)
        y = self.memory.retrieve(q, state)

        return y

    def update(
        self, x: torch.Tensor, state: MemoryState | None = None
    ) -> tuple[torch.Tensor, MemoryState]:
        B, L, _ = x.shape
        device = x.device
        if state is None:
            state = self.memory.init_state(B, device=device)

        y_list = []
        current_state = state
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            x_chunk = x[:, start:end, :]  # (B, c, d)

            k = self.Wk(x_chunk)
            k = self.conv_k(k)

            v = self.Wv(x_chunk)
            v = self.conv_v(v)

            eta, alpha, theta = self._hyper(x_chunk)  # (B, c, 1, 1)

            v_hat, current_state, _loss = self.memory.update(
                k=k, v=v, state=current_state, eta=eta, alpha=alpha, theta=theta
            )
            y_list.append(v_hat)  # (B, c, d)

        y = torch.cat(y_list, dim=1)  # (B, L, d)
        return (y, current_state)
