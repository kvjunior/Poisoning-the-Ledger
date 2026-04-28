"""
Unified fraud detector: wraps any GNN, handles training, evaluation,
multi-seed runs, and class imbalance.  Also supports defense models
(RobustGNN, ProvenanceGNN) through a generic interface.
"""
import sys, copy, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from typing import Dict, Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from models.gnns import MODEL_REGISTRY


class FraudDetector:
    def __init__(self, model_type="gat", num_features=166, num_classes=2,
                 custom_model=None, device=None):
        self.device = device or CFG.device
        self.model_type = model_type
        self.num_features = num_features

        if custom_model is not None:
            self.model = custom_model.to(self.device)
        else:
            C = CFG.model
            kw = dict(in_dim=num_features, hid=C.hidden_dim,
                      out=num_classes, layers=C.num_layers, drop=C.dropout)
            if model_type == "gat":
                kw["heads"] = C.heads
            if model_type == "sage":
                kw["aggr"] = C.aggr
            self.model = MODEL_REGISTRY[model_type](**kw).to(self.device)

        self.history: List[dict] = []
        self._best_state = None
        # Flag: does model.forward need timesteps?
        self._needs_timestep = hasattr(self.model, "needs_timestep") and self.model.needs_timestep

    # ------------------------------------------------------------------
    def fit(self, data: Data, epochs=None, lr=None, patience=None,
            verbose=True) -> Dict:
        T = CFG.train
        epochs = epochs or T.epochs
        lr = lr or T.lr
        patience = patience or T.patience

        data = data.to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=T.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # class weights
        weight = self._class_weight(data)

        best_val = float("inf")
        wait = 0
        self.history = []

        for ep in range(1, epochs + 1):
            self.model.train()
            opt.zero_grad()
            out = self._forward(data)
            mask = data.train_mask & (data.y >= 0)
            loss = F.cross_entropy(out[mask], data.y[mask], weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()

            # validate
            vl, vm = self._eval_split(data, data.val_mask)
            self.history.append(dict(epoch=ep, train_loss=loss.item(),
                                     val_loss=vl, val_f1=vm["f1"]))
            if vl < best_val:
                best_val = vl
                self._best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if verbose and ep % 50 == 0:
                print(f"  ep {ep:3d} | loss {loss.item():.4f} "
                      f"| val_f1 {vm['f1']:.4f}")
            if wait >= patience:
                if verbose:
                    print(f"  early stop @ ep {ep}")
                break

        if self._best_state:
            self.model.load_state_dict(self._best_state)

        _, tm = self._eval_split(data, data.test_mask)
        if verbose:
            print(f"  TEST  F1={tm['f1']:.4f}  P={tm['prec']:.4f}  "
                  f"R={tm['rec']:.4f}")
        return tm

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        return self._forward(data).argmax(1).cpu()

    @torch.no_grad()
    def predict_proba(self, data: Data) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        return F.softmax(self._forward(data), dim=1).cpu()

    def evaluate(self, data: Data, split="test") -> Dict:
        data = data.to(self.device)
        _, m = self._eval_split(data, getattr(data, f"{split}_mask"))
        return m

    @torch.no_grad()
    def get_embeddings(self, data: Data) -> torch.Tensor:
        data = data.to(self.device)
        self.model.eval()
        return self.model.get_embeddings(data.x, data.edge_index).cpu()

    # ------------------------------------------------------------------
    def _forward(self, data):
        if self._needs_timestep:
            ts = data.timestep if hasattr(data, "timestep") else None
            return self.model(data.x, data.edge_index, ts)
        return self.model(data.x, data.edge_index)

    def _class_weight(self, data):
        mask = data.train_mask & (data.y >= 0)
        labels = data.y[mask]
        n0 = (labels == 0).sum().float()
        n1 = (labels == 1).sum().float()
        if n1 > 0:
            w = min(n0 / n1, CFG.train.class_weight_illicit)
            return torch.tensor([1.0, w], device=self.device)
        return None

    @torch.no_grad()
    def _eval_split(self, data, mask):
        self.model.eval()
        out = self._forward(data)
        m = mask & (data.y >= 0)
        if m.sum() == 0:
            return 0., dict(f1=0, prec=0, rec=0, acc=0, tp=0, fp=0, fn=0, tn=0)
        loss = F.cross_entropy(out[m], data.y[m]).item()
        pred = out[m].argmax(1)
        lab = data.y[m]
        tp = ((pred == 1) & (lab == 1)).sum().float()
        fp = ((pred == 1) & (lab == 0)).sum().float()
        fn = ((pred == 0) & (lab == 1)).sum().float()
        tn = ((pred == 0) & (lab == 0)).sum().float()
        p = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.
        r = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.
        acc = ((tp + tn) / (tp + fp + fn + tn)).item()
        return loss, dict(f1=f1, prec=p, rec=r, acc=acc,
                          tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    def save(self, path):
        torch.save(dict(state=self.model.state_dict(),
                        model_type=self.model_type,
                        history=self.history), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state"])

    def reset(self):
        """Reinitialize all weights for fresh training."""
        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.model.apply(_reset)
