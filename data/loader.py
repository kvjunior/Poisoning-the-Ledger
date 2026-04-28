"""
Dataset loading for Elliptic Bitcoin and synthetic graphs.

Elliptic: 203,769 nodes, 234,355 edges, 49 timesteps.
Classes: 1=illicit, 2=licit, unknown.  Mapped to 1, 0, -1 internally.
"""
import os, sys, numpy as np, pandas as pd, torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG


def load_elliptic(root: str = None) -> Data:
    root = root or CFG.data.elliptic_dir
    feat_path = os.path.join(root, "elliptic_txs_features.csv")
    edge_path = os.path.join(root, "elliptic_txs_edgelist.csv")
    cls_path = os.path.join(root, "elliptic_txs_classes.csv")

    for p in (feat_path, edge_path, cls_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Download the Elliptic dataset from "
                "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
                f"and extract to {root}/"
            )

    # ---- features ----
    feat_df = pd.read_csv(feat_path, header=None)
    node_ids = feat_df.iloc[:, 0].values
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    x = torch.tensor(feat_df.iloc[:, 2:].values, dtype=torch.float32)
    timestep = torch.tensor(feat_df.iloc[:, 1].values, dtype=torch.long)

    # ---- edges ----
    edge_df = pd.read_csv(edge_path)
    src_raw, dst_raw = edge_df.iloc[:, 0].values, edge_df.iloc[:, 1].values
    src, dst = [], []
    for s, d in zip(src_raw, dst_raw):
        if s in id2idx and d in id2idx:
            src.append(id2idx[s]); dst.append(id2idx[d])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ---- labels: "1"->illicit(1), "2"->licit(0), "unknown"->(-1) ----
    cls_df = pd.read_csv(cls_path)
    y = torch.full((n,), -1, dtype=torch.long)
    label_map = {"1": 1, "2": 0}
    for _, row in cls_df.iterrows():
        tid, lbl = row["txId"], str(row["class"]).strip()
        if tid in id2idx and lbl in label_map:
            y[id2idx[tid]] = label_map[lbl]

    # ---- splits ----
    train_m, val_m, test_m = _make_splits(y, timestep, n)

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_m, val_mask=val_m, test_mask=test_m,
                timestep=timestep, num_classes=2)
    _print_stats(data)
    return data


def generate_synthetic(n=50000, m=200000, fraud=0.05, d=166, seed=42) -> Data:
    """Controlled synthetic graph for testing when Elliptic is unavailable."""
    rng = np.random.RandomState(seed)
    nf = int(n * fraud)
    nl = n - nf

    # Features: fraud nodes shifted
    x_l = rng.randn(nl, d).astype(np.float32)
    x_f = (rng.randn(nf, d) * 0.8 + 0.4).astype(np.float32)
    x_all = np.vstack([x_l, x_f])
    y_all = np.array([0]*nl + [1]*nf, dtype=np.int64)

    perm = rng.permutation(n)
    x_all, y_all = x_all[perm], y_all[perm]

    # Edges with mild homophily
    fraud_set = set(np.where(y_all == 1)[0])
    src, dst = [], []
    for _ in range(m):
        s = rng.randint(n)
        if s in fraud_set and rng.random() < 0.25:
            pool = list(fraud_set - {s})
            d_node = rng.choice(pool) if pool else rng.randint(n)
        else:
            d_node = rng.randint(n)
        src.append(s); dst.append(d_node)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(x_all, dtype=torch.float32)
    y = torch.tensor(y_all, dtype=torch.long)
    ts = torch.tensor(rng.randint(1, 50, size=n), dtype=torch.long)

    train_m, val_m, test_m = _make_splits(y, ts, n)
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_m, val_mask=val_m, test_mask=test_m,
                timestep=ts, num_classes=2)
    _print_stats(data)
    return data


def _make_splits(y, ts, n):
    labeled = y >= 0
    train_m = torch.zeros(n, dtype=torch.bool)
    val_m = torch.zeros(n, dtype=torch.bool)
    test_m = torch.zeros(n, dtype=torch.bool)

    if CFG.data.use_temporal_split:
        cut = CFG.data.temporal_split_step
        for i in range(n):
            if not labeled[i]:
                continue
            if ts[i].item() <= cut:
                train_m[i] = True
            else:
                test_m[i] = True
        # carve validation from training
        t_idx = train_m.nonzero(as_tuple=True)[0].numpy()
        if len(t_idx) > 10:
            t_labels = y[train_m].numpy()
            tr, va = train_test_split(t_idx, test_size=0.2,
                                      stratify=t_labels, random_state=42)
            train_m[:] = False
            train_m[tr] = True
            val_m[va] = True
    else:
        idx = labeled.nonzero(as_tuple=True)[0].numpy()
        lbl = y[labeled].numpy()
        tr, rest, _, rest_l = train_test_split(idx, lbl, test_size=0.4,
                                               stratify=lbl, random_state=42)
        va, te = train_test_split(rest, test_size=0.5,
                                  stratify=rest_l, random_state=42)
        train_m[tr] = True; val_m[va] = True; test_m[te] = True

    return train_m, val_m, test_m


def _print_stats(data):
    y = data.y
    print(f"[Data] nodes={data.num_nodes}  edges={data.edge_index.shape[1]}  "
          f"feats={data.x.shape[1]}")
    print(f"[Data] licit={(y==0).sum()}  illicit={(y==1).sum()}  "
          f"unknown={(y==-1).sum()}")
    print(f"[Data] train={data.train_mask.sum()}  val={data.val_mask.sum()}  "
          f"test={data.test_mask.sum()}")
