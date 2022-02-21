import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.model import MultiAdjGNN, GCN
from TCN.model import GatedTemporalConv
from STNET.layers import EncoderDecoder, Encoder, Decoder
from STNET.utils import get_dilation
from libs.utils import clones, transform_adj


class BaseSupports(nn.Module):
    """Some Information about AdaptiveAdj"""

    def __init__(self, A, adj_type, self_connection=True, mode='bidirection'):
        super(BaseSupports, self).__init__()
        self.mode = mode
        A = transform_adj(A, adj_type)
        if not self_connection:
            A = A.fill_diagonal_(0)
        self.register_buffer('A', F.normalize(A, p=1, dim=1))
        self.register_buffer('AT', F.normalize(A.T, p=1, dim=1))

    def get_supports(self):
        if self.mode == 'bidirection':
            adjs = [self.A, self.AT]
        elif self.mode == 'forward':
            adjs = [self.A]
        elif self.mode == 'backward':
            adjs = [self.AT]
        elif self.mode == 'empty':
            adjs = []
        return adjs

    def num_supports(self):
        return len(self.get_supports())


class Supports(BaseSupports):
    """Some Information about Supports"""

    def __init__(self, A, adj_type, emb_dim=10,
                 adj_mode='bidirection', num_adp=1):
        super().__init__(A, self_connection=True, mode=adj_mode, adj_type=adj_type)
        dim = A.size(0)
        assert dim >= emb_dim
        self.adp_adjs = clones(Adaptive_adj(dim, emb_dim), num_adp)

    def get_supports(self):
        return [*super().get_supports(), *[adp_adj.get_support()
                                           for adp_adj in self.adp_adjs]]


class Adaptive_adj(nn.Module):
    """Some Information about AdaptiveAdj"""

    def __init__(self, N, emb_dim):
        super().__init__()
        assert N >= emb_dim
        self.nodevec1 = nn.Parameter(
            torch.randn(N, emb_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(
            torch.randn(emb_dim, N), requires_grad=True)

    def get_support(self):
        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return adaptive_adj


class GWNET(nn.Module):
    """Some Information about GWNET"""

    def __init__(self, in_dim, enc_in_dim, enc_hid_dim, enc_out_dim, num_enc_blocks, dec_hid_dim, dec_out_dim, kernel_size,
                 num_gnn_layers, num_temp_layers, A, adj_type, adj_mode, num_adaptive_adj, dropout=0.3):
        super(GWNET, self).__init__()
        adaptive_adj = True if num_adaptive_adj > 0 else False
        self.supports = Supports(
            A, emb_dim=10, adj_mode=adj_mode, num_adp=num_adaptive_adj, adj_type=adj_type) if adaptive_adj else BaseSupports(A, mode=adj_mode, adj_type=adj_type)
        self.upscaler = nn.Conv2d(in_dim, enc_in_dim, 1)

        def depth_dependent_temporal_conv(depth):
            return GatedTemporalConv(
                enc_in_dim, enc_hid_dim, enc_out_dim, kernel_size, get_dilation(depth, kernel_size))

        self.net = EncoderDecoder(
            encoder=Encoder(
                temporal_conv=depth_dependent_temporal_conv,
                spatial_conv=MultiAdjGNN(clones(GCN(num_gnn_layers), self.supports.num_supports()),
                                         enc_hid_dim, enc_in_dim, self.supports.num_supports(),
                                         num_gnn_layers, dropout),
                batch_norm=nn.BatchNorm2d(enc_in_dim),
                num_layers=num_temp_layers),
            decoder=Decoder(enc_out_dim, dec_hid_dim, dec_out_dim),
            num_blocks=num_enc_blocks
        )

    def forward(self, x):
        x = self.upscaler(x)
        x = self.net(x, self.supports.get_supports())
        return x.squeeze()
