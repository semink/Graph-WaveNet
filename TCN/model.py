import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """Some Information about TCN"""

    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super(TemporalConv, self).__init__()
        self.padding_size = dilation*(kernel_size-1)
        self.temporal_conv = nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       dilation=(1, dilation))

    def forward(self, x):
        # this padding is important for causality
        x = F.pad(x, (self.padding_size, 0, 0, 0))
        x = self.temporal_conv(x)
        return x


class GatedTemporalConv(nn.Module):
    """Some Information about GatedTCN"""

    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size, dilation):
        super(GatedTemporalConv, self).__init__()
        self.filter = nn.Sequential(
            TemporalConv(in_dim, hidden_dim, kernel_size, dilation), nn.Tanh())
        self.gate = nn.Sequential(
            TemporalConv(in_dim, hidden_dim, kernel_size, dilation), nn.Sigmoid())
        self.skip_conv = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x_filter = self.filter(x)
        x_gate = self.gate(x)
        x = x_filter * x_gate
        skip = self.skip_conv(x)
        return x, skip[..., [-1]]
