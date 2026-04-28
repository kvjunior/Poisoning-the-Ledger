#!/usr/bin/env python3
"""
Train and evaluate clean baselines independently.

Usage:
    python -m experiments.run_baseline --model gat --synthetic
    python -m experiments.run_baseline --model all --seeds 42,123,456,789,1024
"""
import sys, argparse, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from data.loader import load_elliptic, generate_synthetic
from data.preprocessing import normalize_features, graph_stats
from models.detector import FraudDetector
from evaluation.metrics import eval_detector, aggregate_seeds
from utils.logger import save_json


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", default="gat", choices=["gcn","gat","sage","all"])
    pa.add_argument("--synthetic", action="store_true")
    pa.add_argument("--seeds", default="42,123,456,789,1024")
    pa.add_argument("--epochs", type=int, default=None)
    args = pa.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = ["gcn","gat","sage"] if args.model == "all" else [args.model]

    if args.synthetic:
        data = generate_synthetic()
    else:
        data = load_elliptic()
    data = normalize_features(data)

    st = graph_stats(data)
    print("\nGraph statistics:")
    for k, v in st.items():
        print(f"  {k}: {v}")

    all_results = {}
    for mt in models:
        print(f"\n{'='*50}\n  Training {mt.upper()}\n{'='*50}")
        seed_res = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"\n  seed={seed}")
            det = FraudDetector(model_type=mt, num_features=data.x.shape[1])
            kw = {}
            if args.epochs:
                kw["epochs"] = args.epochs
            det.fit(data, verbose=True, **kw)
            m = eval_detector(det, data)
            m["seed"] = seed
            seed_res.append(m)
            det.save(f"{CFG.ckpt_dir}/baseline_{mt}_s{seed}.pt")

        agg = aggregate_seeds(seed_res)
        all_results[mt] = agg
        print(f"\n  {mt.upper()} SUMMARY: F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}  "
              f"P={agg['prec_mean']:.4f}  R={agg['rec_mean']:.4f}  "
              f"AUC={agg['auc_mean']:.4f}")

    save_json(all_results, "baselines.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
