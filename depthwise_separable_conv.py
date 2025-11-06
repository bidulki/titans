from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# 4.4 Architectural Details - 1D depthwise-separable convolution
class CausalDSC1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        pad = self.kernel_size - 1
        if pad > 0:
            x = F.pad(x, (pad, 0))

        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
