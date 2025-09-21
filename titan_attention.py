import torch
from torch import nn
import torch.nn.functional as F
import math


# DepthwiseSeperableConv1d
class DepthwiseSeperableConv1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_dim, 
            input_dim, 
            kernel_size=3, 
            padding=1, 
            groups=input_dim
        )
        self.pointwise = nn.Conv1d(input_dim, output_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class TitanAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Depthwise separable convolutions
        self.q_conv = DepthwiseSeperableConv1d(dim, dim)
        self.k_conv = DepthwiseSeperableConv1d(dim, dim)
        self.v_conv = DepthwiseSeperableConv1d(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        # Project and apply SiLU activation
        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(x))
        v = F.silu(self.v_proj(x))

        # Apply depthwise seperable conv1d
        q = self._apply_conv(q, self.q_conv)
        k = self._apply_conv(k, self.k_conv)
        v = self._apply_conv(v, self.v_conv)

        # L2 normalize queries and keys
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Properly reshape mask for broadcasting
            # mask shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.out_proj(out)

    def _apply_conv(self, tensor, conv: DepthwiseSeperableConv1d):
        conv_input = tensor.transpose(1, 2)
        conv_output = conv(conv_input)
        return conv_output.transpose(1, 2)

