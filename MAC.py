import torch
from torch import nn

from neural_memory import NeuralMemory
from titan_attention import TitanAttention

class MemoryAsContext(nn.Module):
    def __init__(
        self, 
        dim,
        num_heads,
        chunk_size,
        memory_hidden_dim,
        memory_depth,
        num_persistent_tokens,
        num_memory_tokens,
        dropout
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.num_persistent_tokens = num_persistent_tokens
        self.num_memory_tokens = num_memory_tokens

        self.nueral_memory = NeuralMemory(dim, memory_hidden_dim, memory_depth)
        self.attention = TitanAttention(dim, num_heads, dropout)
        self.persistent_memory = nn.Parameter(torch.randn((num_persistent_tokens, dim)))


    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 4.1 Memory as a Context equation(21)
        historical_info = self.nueral_memory.retrieve(x)
        
        # 4.1 Memory as a Context equation(22)
        pm_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([pm_expanded, historical_info, x], dim=1)

        # 4.1 Memory as a Context equation(23)
        y = self.attention(x)

        # 4.1 Memory as a Context equation(24)
        _, new_params = self.nueral_memory.update(y)
        memory_output = self.nueral_memory.retrieve(y)

        return y * memory_output

    def sample(self, sequence):
        chunks = torch.split(sequence, self.chunk_size, dim=1)
        outputs = []
        for chunk in chunks:
            output = self.forward(chunk)
            outputs.append(output[:, -chunk.size(1):])

        return torch.cat(outputs, dim=1)
        
        





