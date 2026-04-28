"""
On-chain poisoning attacks: base class + three strategies.

Attack 1 — Neighborhood Camouflage:
    Surround fraud nodes with licit-looking buffers to dilute GNN aggregation.

Attack 2 — Label-Delay Exploitation:
    Inject borderline nodes during the forensic-labeling lag window;
    label propagation assigns them incorrect "licit" labels.

Attack 3 — Topology Manipulation:
    Reshape fraud-node neighborhoods to match licit structural statistics,
    erasing the topological signal GNNs rely on.

All attacks share the blockchain-specific constraint model:
  - Can ADD nodes (new accounts) and edges (transactions)
  - CANNOT delete or modify existing nodes/edges (immutability)
  - Each injection has a gas cost in USD
"""
import sys, torch, numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import degree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from data.preprocessing import clone_data


# ======================================================================
#  BASE CLASS
# ======================================================================
class PoisoningAttack(ABC):
    def __init__(self, budget_nodes=50, edges_per_node=5, seed=42):
        self.budget_nodes = budget_nodes
        self.edges_per_node = edges_per_node
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.injected_nodes = 0
        self.injected_edges = 0

    @abstractmethod
    def execute(self, data: Data, target_class: int = 1) -> Data:
        ...

    def gas_cost(self):
        A = CFG.attack
        total_gas = (self.injected_nodes + self.injected_edges) * A.gas_per_tx
        eth = total_gas * A.gas_price_gwei / 1e9
        usd = eth * A.eth_price_usd
        return dict(nodes=self.injected_nodes, edges=self.injected_edges,
                    gas=total_gas, eth=eth, usd=usd)

    # --- helpers ---
    def _add_nodes(self, data, feats, labels, ts=None, in_train=True):
        n_new = feats.shape[0]
        data.x = torch.cat([data.x, feats])
        data.y = torch.cat([data.y, labels])
        data.train_mask = torch.cat([data.train_mask,
            torch.full((n_new,), in_train, dtype=torch.bool)])
        data.val_mask = torch.cat([data.val_mask,
            torch.zeros(n_new, dtype=torch.bool)])
        data.test_mask = torch.cat([data.test_mask,
            torch.zeros(n_new, dtype=torch.bool)])
        if hasattr(data, "timestep"):
            if ts is None:
                ts = torch.full((n_new,), data.timestep.max().item(),
                                dtype=torch.long)
            data.timestep = torch.cat([data.timestep, ts])
        self.injected_nodes += n_new
        return data

    def _add_edges(self, data, new_edges):
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        self.injected_edges += new_edges.shape[1]
        return data

    def _sample_licit_features(self, data, k):
        lic = data.x[data.y == 0]
        if len(lic) == 0:
            return torch.randn(k, data.x.shape[1])
        idx = self.rng.choice(len(lic), size=k, replace=True)
        noise = torch.randn(k, data.x.shape[1]) * lic.std(0).unsqueeze(0) * 0.08
        return lic[idx].clone() + noise

    def _fraud_nodes(self, data, cls=1):
        return (data.y == cls).nonzero(as_tuple=True)[0]


# ======================================================================
#  ATTACK 1: NEIGHBORHOOD CAMOUFLAGE
# ======================================================================
class NeighborhoodCamouflage(PoisoningAttack):
    """
    Inject buffer nodes around fraud targets.
    Connected TO the fraud node so GNN aggregation dilutes the fraud signal.
    """
    def execute(self, data, target_class=1):
        poisoned = clone_data(data)
        fraud = self._fraud_nodes(poisoned, target_class)
        if len(fraud) == 0:
            return poisoned

        # allocate budget: protect highest-degree fraud nodes first
        in_d = degree(poisoned.edge_index[1], num_nodes=poisoned.num_nodes)
        scores = in_d[fraud]
        order = scores.argsort(descending=True)
        max_targets = max(1, self.budget_nodes // max(self.edges_per_node, 1))
        targets = fraud[order[:min(max_targets, len(fraud))]]
        bufs_per = max(1, self.budget_nodes // len(targets))

        base = poisoned.num_nodes
        all_feats, src, dst = [], [], []
        licit_pool = (poisoned.y == 0).nonzero(as_tuple=True)[0].numpy()

        for ti, tgt in enumerate(targets):
            tgt = tgt.item()
            feats = self._sample_licit_features(poisoned, bufs_per)
            all_feats.append(feats)
            for b in range(bufs_per):
                bi = base + ti * bufs_per + b
                src.append(bi); dst.append(tgt)  # buffer -> target
                # credibility edges to random licit nodes
                if len(licit_pool) > 0:
                    extra = min(self.edges_per_node - 1, len(licit_pool))
                    for e in self.rng.choice(licit_pool, size=extra, replace=False):
                        src.append(bi); dst.append(int(e))

        feats_t = torch.cat(all_feats)
        labels = torch.zeros(feats_t.shape[0], dtype=torch.long)  # "licit"
        poisoned = self._add_nodes(poisoned, feats_t, labels)
        if src:
            poisoned = self._add_edges(poisoned,
                torch.tensor([src, dst], dtype=torch.long))

        c = self.gas_cost()
        print(f"[Camouflage] +{c['nodes']} nodes, +{c['edges']} edges, "
              f"${c['usd']:.2f}")
        return poisoned


# ======================================================================
#  ATTACK 2: LABEL-DELAY EXPLOITATION
# ======================================================================
class LabelDelayExploit(PoisoningAttack):
    """
    Exploit the window between transaction creation and forensic labeling.
    Inject borderline nodes (feature-space between illicit and licit)
    connected to many licit nodes. Label propagation assigns them "licit",
    creating mislabeled training samples that confuse the detector.
    """
    def __init__(self, budget_nodes=50, edges_per_node=5, seed=42,
                 delay_window=3):
        super().__init__(budget_nodes, edges_per_node, seed)
        self.delay_window = delay_window

    def execute(self, data, target_class=1):
        poisoned = clone_data(data)
        max_ts = poisoned.timestep.max().item() if hasattr(poisoned, "timestep") else 49
        delay_start = max_ts - self.delay_window + 1

        # Step 1: generate borderline features (between illicit and licit)
        feats = self._borderline_features(poisoned, self.budget_nodes)

        # Step 2: place in delay window
        ts = torch.tensor(
            self.rng.randint(delay_start, max_ts + 1, size=self.budget_nodes),
            dtype=torch.long)

        # Step 3: inject as UNKNOWN (-1) initially, NOT in train set
        labels = torch.full((self.budget_nodes,), -1, dtype=torch.long)
        base = poisoned.num_nodes
        poisoned = self._add_nodes(poisoned, feats, labels, ts, in_train=False)

        # Step 4: connect heavily to licit nodes (bidirectional)
        licit_pool = (poisoned.y == 0).nonzero(as_tuple=True)[0].numpy()
        src, dst = [], []
        for i in range(self.budget_nodes):
            ni = base + i
            if len(licit_pool) > 0:
                k = min(self.edges_per_node, len(licit_pool))
                nbrs = self.rng.choice(licit_pool, size=k, replace=False)
                for nb in nbrs:
                    src.append(ni); dst.append(int(nb))
                    src.append(int(nb)); dst.append(ni)
        if src:
            poisoned = self._add_edges(poisoned,
                torch.tensor([src, dst], dtype=torch.long))

        # Step 5: simple label propagation (majority of 1-hop neighbors)
        propagated = self._propagate(poisoned)

        # Step 6: assign propagated labels, add to training set
        success = 0
        for i in range(self.budget_nodes):
            ni = base + i
            poisoned.y[ni] = propagated[ni]
            poisoned.train_mask[ni] = True
            if propagated[ni] == 0:
                success += 1

        c = self.gas_cost()
        print(f"[LabelDelay] +{c['nodes']} nodes, +{c['edges']} edges, "
              f"${c['usd']:.2f}, mislabeled={success}")
        return poisoned

    def _borderline_features(self, data, k):
        lic = data.x[data.y == 0]
        ill = data.x[data.y == 1]
        if len(ill) == 0 or len(lic) == 0:
            return self._sample_licit_features(data, k)
        mu_l, mu_i = lic.mean(0), ill.mean(0)
        center = mu_i + 0.6 * (mu_l - mu_i)  # 60% toward licit
        std_i = ill.std(0).clamp(min=1e-6)
        return center.unsqueeze(0) + torch.randn(k, data.x.shape[1]) * std_i * 0.25

    def _propagate(self, data, iterations=10):
        """Iterative 1-hop majority vote for unknown nodes."""
        labels = data.y.clone()
        src, dst = data.edge_index
        for _ in range(iterations):
            new = labels.clone()
            unknown = (labels == -1).nonzero(as_tuple=True)[0]
            for node in unknown:
                n = node.item()
                mask = (src == n) | (dst == n)
                nbrs = torch.cat([src[mask], dst[mask]])
                nbrs = nbrs[nbrs != n].unique()
                if len(nbrs) == 0:
                    continue
                nl = labels[nbrs]
                known = nl[nl >= 0]
                if len(known) == 0:
                    continue
                new[n] = 0 if (known == 0).sum() >= (known == 1).sum() else 1
            if (new == labels).all():
                break
            labels = new
        return labels


# ======================================================================
#  ATTACK 3: TOPOLOGY MANIPULATION
# ======================================================================
class TopologyManipulation(PoisoningAttack):
    """
    Normalize fraud-node neighborhoods to match licit structural statistics.
    Eliminates the degree/clustering differences GNNs exploit.
    """
    def execute(self, data, target_class=1):
        poisoned = clone_data(data)
        in_d = degree(poisoned.edge_index[1], num_nodes=poisoned.num_nodes)
        out_d = degree(poisoned.edge_index[0], num_nodes=poisoned.num_nodes)
        total = in_d + out_d

        lic_mask = poisoned.y == 0
        ill_mask = poisoned.y == 1
        if not ill_mask.any() or not lic_mask.any():
            return poisoned

        target_deg = total[lic_mask].mean().item()
        fraud = ill_mask.nonzero(as_tuple=True)[0]

        # sort by deviation from licit average (most anomalous first)
        devs = (total[fraud] - target_deg).abs()
        order = devs.argsort(descending=True)

        base = poisoned.num_nodes
        all_feats, src, dst = [], [], []
        used = 0
        licit_pool = lic_mask.nonzero(as_tuple=True)[0].numpy()

        for idx in order:
            if used >= self.budget_nodes:
                break
            fn = fraud[idx].item()
            cur = total[fn].item()
            if cur >= target_deg:
                continue
            need = min(int(target_deg - cur), self.budget_nodes - used)
            if need <= 0:
                continue

            feats = self._sample_licit_features(poisoned, need)
            all_feats.append(feats)

            for b in range(need):
                bi = base + used + b
                # bidirectional to pad degree
                src.append(bi); dst.append(fn)
                src.append(fn); dst.append(bi)
                # inter-buffer clustering
                if b > 0 and self.rng.random() < 0.15:
                    src.append(bi); dst.append(base + used + b - 1)
                    src.append(base + used + b - 1); dst.append(bi)
                # credibility to licit pool
                if len(licit_pool) > 0:
                    ex = min(self.edges_per_node - 2, len(licit_pool))
                    if ex > 0:
                        for e in self.rng.choice(licit_pool, ex, replace=False):
                            src.append(bi); dst.append(int(e))
            used += need

        if all_feats:
            ft = torch.cat(all_feats)
            labels = torch.zeros(ft.shape[0], dtype=torch.long)
            max_ts = poisoned.timestep.max().item() if hasattr(poisoned, "timestep") else 49
            ts = torch.full((ft.shape[0],), max_ts, dtype=torch.long)
            poisoned = self._add_nodes(poisoned, ft, labels, ts)
        if src:
            poisoned = self._add_edges(poisoned,
                torch.tensor([src, dst], dtype=torch.long))

        # verify
        new_total = degree(poisoned.edge_index[1], num_nodes=poisoned.num_nodes) + \
                    degree(poisoned.edge_index[0], num_nodes=poisoned.num_nodes)
        new_avg = new_total[ill_mask[:data.num_nodes] if len(ill_mask) > poisoned.num_nodes
                            else ill_mask].mean().item() if ill_mask.any() else 0

        c = self.gas_cost()
        print(f"[Topology] +{c['nodes']} nodes, +{c['edges']} edges, "
              f"${c['usd']:.2f}, illicit avg deg: {new_avg:.1f} (target: {target_deg:.1f})")
        return poisoned


ATTACK_REGISTRY = {
    "camouflage": NeighborhoodCamouflage,
    "label_delay": LabelDelayExploit,
    "topology": TopologyManipulation,
}
