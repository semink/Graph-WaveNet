import torch
import torch.nn as nn


class MultiAdjGNN(nn.Module):
    def __init__(self, gnns, in_dim, out_dim, num_supports, order, dropout):
        super(MultiAdjGNN, self).__init__()
        assert len(gnns) == num_supports
        self.gnns = gnns
        self.linear = nn.Conv2d(
            in_dim * (order * num_supports + 1), out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        x = torch.cat([x,  # add GNN with zero order
                       *[gnn(x, A) for gnn, A in zip(self.gnns, adjs)]], dim=1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    """Some Information about GCN"""

    def __init__(self, order, operation='ncvl, vw -> ncwl'):
        # follow the original implementation. The correct operation should be
        # 'ncvl, wv -> ncwl' but performs worse (don't know why).
        super(GCN, self).__init__()
        self.order = order
        self.conv_operation = operation

    def gconv(self, x, A):
        return torch.einsum(self.conv_operation, x, A)

    def forward(self, x, A):
        x = torch.cat([self.gconv(x, torch.matrix_power(A, k + 1))
                       for k in range(self.order)], dim=1)
        return x
