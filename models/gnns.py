"""
GNN architectures: GCN, GAT, GraphSAGE.
All share the same forward/get_embeddings interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm


class _BaseGNN(nn.Module):
    """Common skeleton: stack of conv+bn+act layers, final linear."""

    def __init__(self, convs, bns, num_layers, dropout, skip_proj=None):
        super().__init__()
        self.convs = convs
        self.bns = bns
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip_proj = skip_proj

    def forward(self, x, edge_index):
        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h) if not isinstance(self.convs[i], GATConv) else F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if i == 0 and self.skip_proj is not None:
                h = h + self.skip_proj(x)
        return self.convs[-1](h, edge_index)

    def get_embeddings(self, x, edge_index):
        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
        return h


class GCN(_BaseGNN):
    def __init__(self, in_dim, hid, out, layers=3, drop=0.3):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        convs.append(GCNConv(in_dim, hid))
        bns.append(BatchNorm(hid))
        for _ in range(layers - 2):
            convs.append(GCNConv(hid, hid))
            bns.append(BatchNorm(hid))
        convs.append(GCNConv(hid, out))
        skip = nn.Linear(in_dim, hid) if in_dim != hid else None
        super().__init__(convs, bns, layers, drop, skip)


class GAT(_BaseGNN):
    def __init__(self, in_dim, hid, out, layers=3, drop=0.3, heads=4):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        convs.append(GATConv(in_dim, hid // heads, heads=heads, dropout=drop))
        bns.append(BatchNorm(hid))
        for _ in range(layers - 2):
            convs.append(GATConv(hid, hid // heads, heads=heads, dropout=drop))
            bns.append(BatchNorm(hid))
        convs.append(GATConv(hid, out, heads=1, concat=False, dropout=drop))
        skip = nn.Linear(in_dim, hid) if in_dim != hid else None
        super().__init__(convs, bns, layers, drop, skip)


class SAGE(_BaseGNN):
    def __init__(self, in_dim, hid, out, layers=3, drop=0.3, aggr="mean"):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        convs.append(SAGEConv(in_dim, hid, aggr=aggr))
        bns.append(BatchNorm(hid))
        for _ in range(layers - 2):
            convs.append(SAGEConv(hid, hid, aggr=aggr))
            bns.append(BatchNorm(hid))
        convs.append(SAGEConv(hid, out, aggr=aggr))
        skip = nn.Linear(in_dim, hid) if in_dim != hid else None
        super().__init__(convs, bns, layers, drop, skip)


MODEL_REGISTRY = {"gcn": GCN, "gat": GAT, "sage": SAGE}
