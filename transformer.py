import torch
from torch import nn
import torch.nn.functional as F


# DepthwiseSeperableConv1d (V)
# Reference: 4.4 Architectural Details - Convolution
class DepthwiseSeperableConv1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, 
            padding=1, groups=input_dim
        )
        self.pointwise = nn.Conv1d(input_dim, output_dim, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# GatingMechanism (?)
# Reference: 4.4 Architectural Details - Gating
class GatingMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.transform_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        transformed = self.transform_proj(x)
        return gate * transformed


class TitanAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
    
        # Projections for Q, K, V
        # Reference: 4.4 Architectural Details - Convolution
        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        self.k_proj = nn.Linear(dim, num_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_heads * head_dim)

        # Depthwise separable convolutions
        # Reference: 4.4 Architectural Details - Convolution
        self.q_conv = DepthwiseSeperableConv1d(
            num_heads * head_dim,
            num_heads * head_dim
        )

        self.k_conv = DepthwiseSeperableConv1d(
            num_heads * head_dim,
            num_heads * head_dim
        )

        self.v_conv = DepthwiseSeperableConv1d(
            num_heads * head_dim,
            num_heads * head_dim
        )

        self.out_proj = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        # Project and apply SiLU activation
        # Reference: 4.4 Architectural Details - 1 Paragraph(SiLU)
        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(x))
        v = F.silu(self.v_proj(x))

        # Reshape for depthwise conv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply depthwise separable convolutions
        q = self.q_conv(q).transpose(1, 2)
        k = self.k_conv(k).transpose(1, 2)
        v = self.v_conv(v).transpose(1, 2)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L2 normalize queries and keys
        # Reference: 4.4 Architectural Details - 1 Paragraph(l2-norm)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Properly reshape mask for broadcasting
            # mask shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out

class TitanBlock(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TitanAttention(dim, num_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            GatingMechanism(mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Residual connection for attention
        # Reference: 4.4 Architectural Details - 1 Paragraph(residual connections)
        x = x + self.attn(self.norm1(x), mask)
        # Residual connection for MLP
        x = x + self.mlp(self.norm2(x))
        return x

class TitanTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        head_dim=64,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TitanBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

