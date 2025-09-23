import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_mlp import MemoryMLP, MemoryState
from utils import make_linear


# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(self, dim, memory_hidden_dim, memory_depth):
        super().__init__()
        self.dim = dim

        self.memory = MemoryMLP(dim, memory_hidden_dim, memory_depth)

        self.Wk = make_linear(dim, dim)
        self.Wv = make_linear(dim, dim)

        self.hyper = nn.Linear(dim, 3)

        # self.alpha = 0.001
        # self.eta = 0.60
        # self.theta = 0.05

    def _hyper(self, x: torch.Tensor):
        h = self.hyper(x)
        eta = torch.sigmoid(h[:, 0:1]).unsqueeze(-1)
        alpha = torch.sigmoid(h[:, 1:2]).unsqueeze(-1)
        theta = (F.softplus(h[:, 2:3]) * 1e-2 + 1e-6).unsqueeze(-1)
        return eta, alpha, theta

    def forward(
        self, x: torch.Tensor, state: MemoryState | None = None
    ) -> tuple[torch.Tensor, MemoryState]:
        B, L, _ = x.shape
        device = x.device
        if state is None:
            state = self.memory.init_state(B, device=device)

        y_list = []
        current_state = state
        for t in range(L):
            x = x[:, t, :]
            k = self.Wk(x)
            v = self.Wv(x)
            eta, alpha, theta = self._hyper(x)

            v_hat, current_state, _loss = self.memory.step(
                k=k, v=v, state=current_state, eta=eta, alpha=alpha, theta=theta
            )
            y_list.append(v_hat)

        y = torch.stack(y_list, dim=1)
        return (y, current_state)
