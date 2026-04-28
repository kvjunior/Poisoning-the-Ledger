"""Logging and result serialization."""
import os, json, torch, numpy as np
from config import CFG

def save_json(obj, name):
    path = os.path.join(CFG.output_dir, name)
    with open(path, "w") as f:
        json.dump(_conv(obj), f, indent=2)
    print(f"[saved] {path}")

def _conv(o):
    if isinstance(o, dict):
        return {k: _conv(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_conv(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.tolist()
    return o
