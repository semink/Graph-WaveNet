import torch.nn as nn
from copy import deepcopy as c
from libs.utils import clones


def skip_connection(modules, x, adjs):
    skip = 0
    for module in modules:
        x, s = module(x, adjs)
        skip = skip + s
    return x, skip


class EncoderDecoder(nn.Module):
    """Some Information about EncoderDecoder"""

    def __init__(self, encoder, decoder, num_blocks):
        super(EncoderDecoder, self).__init__()
        self.encoder_blocks = clones(encoder, num_blocks)
        self.decoder = decoder

    def forward(self, x, adjs):
        x, skip = skip_connection(self.encoder_blocks, x, adjs)
        x = self.decoder(skip)
        return x


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(self, temporal_conv, spatial_conv, batch_norm, num_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(c(spatial_conv),c(temporal_conv(depth)),c(batch_norm)) for depth in range(num_layers)])

    def forward(self, x, adjs):
        x, skip = skip_connection(self.encoder_layers, x, adjs)
        return x, skip


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(in_dim, hid_dim, 1)
        self.linear2 = nn.Conv2d(hid_dim, out_dim, 1)

    def forward(self, x):
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Some Information about EncoderLayer"""

    def __init__(self, gnn, gated_temporal_conv, batch_norm):
        super(EncoderLayer, self).__init__()
        self.gnn = gnn
        self.gated_temporal_conv = gated_temporal_conv
        self.batch_norm = batch_norm

    def forward(self, x, adjs):
        res = x
        x, skip = self.gated_temporal_conv(x)
        x = self.gnn(x, adjs)
        x = res + x
        x = self.batch_norm(x)
        return x, skip
