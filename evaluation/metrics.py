"""
Evaluation: metrics, attack success rate, amplification, economics.
Bootstrap confidence intervals for multi-seed aggregation.
"""
import sys, torch, numpy as np, pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG


def full_metrics(y_true, y_pred, y_prob=None) -> Dict:
    m = dict(
        f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        prec=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        rec=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        acc=(y_true == y_pred).mean(),
    )
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["auc"] = roc_auc_score(y_true, y_prob)
    else:
        m["auc"] = 0.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m["tn"], m["fp"] = int(cm[0, 0]), int(cm[0, 1])
    m["fn"], m["tp"] = int(cm[1, 0]), int(cm[1, 1])
    return m


def eval_detector(det, data, split="test") -> Dict:
    """Run detector on a split, return full metrics."""
    data_d = data.to(det.device)
    preds = det.predict(data_d).numpy()
    proba = det.predict_proba(data_d)[:, 1].numpy()
    mask = getattr(data, f"{split}_mask") & (data.y >= 0)
    if mask.sum() == 0:
        return {k: 0. for k in ("f1","prec","rec","mcc","acc","auc",
                                 "tn","fp","fn","tp")}
    return full_metrics(data.y[mask].numpy(), preds[mask], proba[mask])


def attack_success_rate(clean_pred, poison_pred, data, cls=1) -> Dict:
    """Fraction of correctly-detected fraud that becomes misclassified."""
    mask = (data.y == cls) & data.test_mask
    if mask.sum() == 0:
        return dict(asr=0., flipped=0, detected=0, total=mask.sum().item())
    correct = clean_pred[mask] == cls
    n_det = correct.sum().item()
    if n_det == 0:
        return dict(asr=0., flipped=0, detected=0, total=mask.sum().item())
    # how many of those are now wrong?
    n_min = min(len(clean_pred), len(poison_pred))
    mask_trim = mask[:n_min]
    flipped = ((clean_pred[:n_min][mask_trim] == cls) &
               (poison_pred[:n_min][mask_trim] != cls)).sum().item()
    return dict(asr=flipped / n_det, flipped=flipped,
                detected=n_det, total=mask.sum().item())


def amplification_factor(clean_pred, poison_pred, data,
                         inject_start, hops=2) -> Dict:
    """How many original nodes change prediction per injected node."""
    n_inj = data.num_nodes - inject_start
    if n_inj == 0:
        return dict(amp=0., changed=0, in_radius=0, injected=0)
    # BFS to find k-hop neighborhood of injected nodes
    affected = set()
    frontier = set(range(inject_start, data.num_nodes))
    src, dst = data.edge_index
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            m = (src == n) | (dst == n)
            nxt.update(torch.cat([src[m], dst[m]]).unique().tolist())
        affected.update(nxt)
        frontier = nxt - affected
    orig = {n for n in affected if n < inject_start}
    n_min = min(len(clean_pred), inject_start)
    changed = sum(1 for n in orig if n < n_min and clean_pred[n] != poison_pred[n])
    return dict(amp=changed / max(n_inj, 1), changed=changed,
                in_radius=len(orig), injected=n_inj)


def aggregate_seeds(results_per_seed: List[Dict]) -> Dict:
    """Aggregate metrics across seeds with mean ± std."""
    keys = [k for k in results_per_seed[0] if isinstance(results_per_seed[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_per_seed]
        agg[f"{k}_mean"] = np.mean(vals)
        agg[f"{k}_std"] = np.std(vals)
        agg[f"{k}_ci95"] = 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)
    return agg


# --- Economics ---
def gas_to_usd(nodes, edges, gas_gwei=None, eth_usd=None):
    A = CFG.attack
    gp = gas_gwei or A.gas_price_gwei
    ep = eth_usd or A.eth_price_usd
    gas = (nodes + edges) * A.gas_per_tx
    eth = gas * gp / 1e9
    return dict(gas=gas, eth=eth, usd=eth * ep)


def cost_effectiveness_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        c = gas_to_usd(r.get("nodes_inj", 0), r.get("edges_inj", 0))
        rows.append(dict(
            attack=r.get("attack", ""),
            budget=r.get("budget", 0),
            nodes=r.get("nodes_inj", 0),
            edges=r.get("edges_inj", 0),
            usd=c["usd"],
            f1_clean=r.get("f1_clean", 0),
            f1_poison=r.get("f1_poison", 0),
            f1_drop=r.get("f1_clean", 0) - r.get("f1_poison", 0),
            f1_drop_pct=((r.get("f1_clean", 0) - r.get("f1_poison", 0))
                         / max(r.get("f1_clean", 1e-6), 1e-6) * 100),
            asr=r.get("asr", 0),
            amp=r.get("amp", 0),
        ))
    return pd.DataFrame(rows)
