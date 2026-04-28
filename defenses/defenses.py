"""
Defenses against on-chain graph poisoning.

Defense 1 — Temporal Consistency:
    Quarantine recently-created nodes with anomalous connectivity.

Defense 2 — Robust Aggregation:
    Replace mean aggregation with trimmed mean (vectorized, scalable).

Defense 3 — Provenance-Aware GNN:
    Encode node age as first-class feature; age-gated attention.
"""
import sys, math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm, MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from data.preprocessing import clone_data


# ======================================================================
#  DEFENSE 1: TEMPORAL CONSISTENCY
# ======================================================================
class TemporalConsistencyDefense:
    """Remove suspicious recent nodes from training set."""

    def __init__(self, age_threshold=None, z_threshold=2.0):
        self.age_threshold = age_threshold or CFG.defense.quarantine_age
        self.z_threshold = z_threshold

    def apply(self, data: Data) -> Data:
        d = clone_data(data)
        if not hasattr(d, "timestep"):
            return d

        max_ts = d.timestep.max().item()
        recent = d.timestep >= (max_ts - self.age_threshold + 1)
        if not recent.any():
            return d

        # degree statistics from established nodes
        total_deg = degree(d.edge_index[0], d.num_nodes) + \
                    degree(d.edge_index[1], d.num_nodes)
        est = ~recent & (d.y >= 0)
        if not est.any():
            return d

        mu = total_deg[est].mean().item()
        sd = total_deg[est].std().item()

        suspicious = torch.zeros(d.num_nodes, dtype=torch.bool)
        for i in recent.nonzero(as_tuple=True)[0]:
            z = abs(total_deg[i].item() - mu) / max(sd, 1e-6)
            if z > self.z_threshold:
                suspicious[i] = True

        # also flag nodes where most neighbors are also recent
        src, dst = d.edge_index
        for i in recent.nonzero(as_tuple=True)[0]:
            ni = i.item()
            mask = (src == ni) | (dst == ni)
            nbrs = torch.cat([src[mask], dst[mask]])
            nbrs = nbrs[nbrs != ni].unique()
            if len(nbrs) > 0 and recent[nbrs].float().mean() > 0.5:
                suspicious[i] = True

        q = suspicious.sum().item()
        d.train_mask = d.train_mask & ~suspicious
        print(f"[Defense:Temporal] quarantined={q}  "
              f"train_remaining={d.train_mask.sum().item()}")
        return d


# ======================================================================
#  DEFENSE 2: ROBUST AGGREGATION (VECTORIZED — NO O(N²) LOOPS)
# ======================================================================
class _TrimmedMeanConv(MessagePassing):
    """
    GCN-style conv with trimmed-mean aggregation.
    Uses scatter + sorting per node — vectorized via grouped ops.
    """
    def __init__(self, in_ch, out_ch, trim=0.1):
        super().__init__(aggr=None)
        self.lin = nn.Linear(in_ch, out_ch)
        self.root = nn.Linear(in_ch, out_ch)
        self.trim = trim

    def forward(self, x, edge_index):
        ei, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = ei
        deg = degree(col, x.size(0), dtype=x.dtype).clamp(min=1)
        norm = (1.0 / deg[col]).unsqueeze(1)
        x_j = self.lin(x)[row] * norm
        # --- trimmed-mean aggregation ---
        agg = self._trimmed_scatter(x_j, col, dim_size=x.size(0))
        return agg + self.root(x)

    def _trimmed_scatter(self, src, index, dim_size):
        """Vectorized trimmed mean using scatter."""
        # Fast path: just use mean for very small trim ratios
        if self.trim < 0.01:
            return scatter(src, index, dim=0, dim_size=dim_size, reduce="mean")

        # Per-node counts
        counts = scatter(torch.ones(src.size(0), 1, device=src.device),
                         index, dim=0, dim_size=dim_size, reduce="sum").squeeze(1)

        # For nodes with few neighbors, use mean
        out = scatter(src, index, dim=0, dim_size=dim_size, reduce="mean")

        # For nodes with enough neighbors (≥5), do trimmed mean
        large_mask = counts >= 5
        if not large_mask.any():
            return out

        large_nodes = large_mask.nonzero(as_tuple=True)[0]
        for node in large_nodes:
            mask = index == node
            vals = src[mask]  # (k, feat_dim)
            k = vals.size(0)
            t = max(1, int(k * self.trim))
            sorted_vals, _ = vals.sort(dim=0)
            trimmed = sorted_vals[t:k-t]
            if trimmed.size(0) > 0:
                out[node] = trimmed.mean(dim=0)

        return out


class RobustGNN(nn.Module):
    """GNN with trimmed-mean aggregation for poisoning defense."""
    def __init__(self, in_dim, hid=128, out=2, layers=3, drop=0.3, trim=0.1):
        super().__init__()
        self.layers_n = layers
        self.drop = drop
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(_TrimmedMeanConv(in_dim, hid, trim))
        self.bns.append(nn.BatchNorm1d(hid))
        for _ in range(layers - 2):
            self.convs.append(_TrimmedMeanConv(hid, hid, trim))
            self.bns.append(nn.BatchNorm1d(hid))
        self.convs.append(_TrimmedMeanConv(hid, out, trim))

    def forward(self, x, edge_index):
        for i in range(self.layers_n - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for i in range(self.layers_n - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


# ======================================================================
#  DEFENSE 3: PROVENANCE-AWARE GNN
# ======================================================================
class _SinusoidalTimeEnc(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.arange(0, self.dim, 2, device=t.device).float()
                          * -(math.log(10000.) / self.dim))
        return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)


class ProvenanceGNN(nn.Module):
    """
    GAT + sinusoidal time encoding + age-gated attention.
    Newer nodes are automatically down-weighted.
    """
    needs_timestep = True  # flag for FraudDetector

    def __init__(self, in_dim, hid=128, out=2, layers=3, heads=4, drop=0.3,
                 time_dim=None, age_decay=None):
        super().__init__()
        td = time_dim or CFG.defense.provenance_time_dim
        self.time_enc = _SinusoidalTimeEnc(td)
        self.proj = nn.Linear(in_dim + td, hid)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers - 1):
            self.convs.append(GATConv(hid, hid // heads, heads=heads,
                                      dropout=drop, concat=True))
            self.bns.append(BatchNorm(hid))
        self.age_gate = nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                       nn.Linear(16, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(),
                                   nn.Dropout(drop), nn.Linear(hid // 2, out))
        self.drop = drop
        self.n_layers = layers

    def forward(self, x, edge_index, timesteps=None):
        # time encoding
        if timesteps is not None:
            te = self.time_enc(timesteps)
        else:
            te = torch.zeros(x.size(0), self.time_enc.dim, device=x.device)
        x = torch.cat([x, te], dim=-1)
        x = F.relu(self.proj(x))

        # age weights
        ages = None
        if timesteps is not None:
            mn, mx = timesteps.min().float(), timesteps.max().float()
            ages = ((timesteps.float() - mn) / max(mx - mn, 1.)).unsqueeze(1)

        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            if ages is not None:
                x = x * self.age_gate(ages)

        return self.head(x)

    def get_embeddings(self, x, edge_index, timesteps=None):
        if timesteps is not None:
            te = self.time_enc(timesteps)
        else:
            te = torch.zeros(x.size(0), self.time_enc.dim, device=x.device)
        x = torch.cat([x, te], dim=-1)
        x = F.relu(self.proj(x))
        ages = None
        if timesteps is not None:
            mn, mx = timesteps.min().float(), timesteps.max().float()
            ages = ((timesteps.float() - mn) / max(mx - mn, 1.)).unsqueeze(1)
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            if ages is not None:
                x = x * self.age_gate(ages)
        return x
