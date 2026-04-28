"""Graph preprocessing: normalization, statistics, deep copy."""
import torch, numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import Dict


def normalize_features(data: Data) -> Data:
    mu = data.x.mean(0, keepdim=True)
    sd = data.x.std(0, keepdim=True).clamp(min=1e-8)
    data.x = (data.x - mu) / sd
    data._feat_mu, data._feat_sd = mu, sd
    return data


def graph_stats(data: Data) -> Dict:
    y = data.y
    in_d = degree(data.edge_index[1], num_nodes=data.num_nodes)
    out_d = degree(data.edge_index[0], num_nodes=data.num_nodes)
    lab = y >= 0
    ill = y == 1
    lic = y == 0
    return dict(
        nodes=data.num_nodes, edges=data.edge_index.shape[1],
        features=data.x.shape[1],
        labeled=lab.sum().item(), illicit=ill.sum().item(), licit=lic.sum().item(),
        illicit_ratio=ill.sum().item() / max(lab.sum().item(), 1),
        avg_in_deg=in_d.mean().item(), avg_out_deg=out_d.mean().item(),
        avg_deg_illicit=in_d[ill].mean().item() if ill.any() else 0,
        avg_deg_licit=in_d[lic].mean().item() if lic.any() else 0,
        timesteps=data.timestep.unique().numel() if hasattr(data, "timestep") else 0,
    )


def clone_data(data: Data) -> Data:
    """Deep copy that preserves all attributes."""
    d = Data(
        x=data.x.clone(), edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        train_mask=data.train_mask.clone(),
        val_mask=data.val_mask.clone(),
        test_mask=data.test_mask.clone(),
    )
    if hasattr(data, "timestep"):
        d.timestep = data.timestep.clone()
    if hasattr(data, "num_classes"):
        d.num_classes = data.num_classes
    return d
