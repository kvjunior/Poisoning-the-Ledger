#!/usr/bin/env python3
"""
Evaluate defenses against a specific attack.

Usage:
    python -m experiments.run_defenses --attack camouflage --defense all --synthetic
    python -m experiments.run_defenses --attack label_delay --defense provenance
"""
import sys, argparse, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from data.loader import load_elliptic, generate_synthetic
from data.preprocessing import normalize_features
from models.detector import FraudDetector
from attacks.attacks import ATTACK_REGISTRY
from defenses.defenses import (TemporalConsistencyDefense, RobustGNN,
                                ProvenanceGNN)
from evaluation.metrics import eval_detector, aggregate_seeds
from utils.logger import save_json


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--attack", default="camouflage",
                    choices=list(ATTACK_REGISTRY.keys()))
    pa.add_argument("--defense", default="all",
                    choices=["no_defense","temporal","robust","provenance","all"])
    pa.add_argument("--budget", type=int, default=100)
    pa.add_argument("--model", default="gat")
    pa.add_argument("--synthetic", action="store_true")
    pa.add_argument("--seeds", default="42,123,456,789,1024")
    args = pa.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    defs = (["no_defense","temporal","robust","provenance"]
            if args.defense == "all" else [args.defense])

    data = generate_synthetic() if args.synthetic else load_elliptic()
    data = normalize_features(data)

    print(f"\nAttack: {args.attack}  Budget: {args.budget}")
    results = {}

    for def_name in defs:
        print(f"\n--- Defense: {def_name} ---")
        seed_res = []

        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)

            # reproduce attack
            kw = dict(budget_nodes=args.budget,
                      edges_per_node=CFG.attack.edges_per_node, seed=seed)
            if args.attack == "label_delay":
                kw["delay_window"] = 3
            poisoned = ATTACK_REGISTRY[args.attack](**kw).execute(
                data, target_class=1)

            # apply defense + train
            if def_name == "no_defense":
                det = FraudDetector(model_type=args.model,
                                   num_features=data.x.shape[1])
                det.fit(poisoned, verbose=False)
                m = eval_detector(det, poisoned)

            elif def_name == "temporal":
                defended = TemporalConsistencyDefense().apply(poisoned)
                det = FraudDetector(model_type=args.model,
                                   num_features=data.x.shape[1])
                det.fit(defended, verbose=False)
                m = eval_detector(det, defended)

            elif def_name == "robust":
                C = CFG.model
                rm = RobustGNN(in_dim=data.x.shape[1], hid=C.hidden_dim,
                               out=2, layers=C.num_layers, drop=C.dropout,
                               trim=CFG.defense.trim_ratio)
                det = FraudDetector(custom_model=rm,
                                   num_features=data.x.shape[1])
                det.fit(poisoned, verbose=False)
                m = eval_detector(det, poisoned)

            elif def_name == "provenance":
                C = CFG.model
                pm = ProvenanceGNN(in_dim=data.x.shape[1], hid=C.hidden_dim,
                                   out=2, layers=C.num_layers,
                                   heads=C.heads, drop=C.dropout)
                det = FraudDetector(custom_model=pm,
                                   num_features=data.x.shape[1])
                det.fit(poisoned, verbose=False)
                m = eval_detector(det, poisoned)

            seed_res.append(m)
            print(f"  seed={seed} | F1={m['f1']:.4f}  P={m['prec']:.4f}  R={m['rec']:.4f}")

        agg = aggregate_seeds(seed_res)
        results[def_name] = dict(f1=agg["f1_mean"], f1_std=agg["f1_std"],
                                  prec=agg["prec_mean"], rec=agg["rec_mean"])
        print(f"  MEAN: F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}")

    save_json(results, f"defense_{args.attack}_{args.defense}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
