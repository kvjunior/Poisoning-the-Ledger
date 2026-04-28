#!/usr/bin/env python3
"""
Run a single attack configuration.

Usage:
    python -m experiments.run_attacks --attack camouflage --budget 100 --synthetic
    python -m experiments.run_attacks --attack all --budget 50 --model gat
"""
import sys, argparse, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from data.loader import load_elliptic, generate_synthetic
from data.preprocessing import normalize_features, clone_data
from models.detector import FraudDetector
from attacks.attacks import ATTACK_REGISTRY
from evaluation.metrics import (eval_detector, attack_success_rate,
                                 amplification_factor, aggregate_seeds)
from utils.logger import save_json


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--attack", default="camouflage",
                    choices=list(ATTACK_REGISTRY.keys()) + ["all"])
    pa.add_argument("--model", default="gat")
    pa.add_argument("--budget", type=int, default=50)
    pa.add_argument("--synthetic", action="store_true")
    pa.add_argument("--seeds", default="42,123,456,789,1024")
    args = pa.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    attacks = list(ATTACK_REGISTRY.keys()) if args.attack == "all" else [args.attack]

    data = generate_synthetic() if args.synthetic else load_elliptic()
    data = normalize_features(data)

    for atk_name in attacks:
        print(f"\n{'='*60}")
        print(f"  Attack: {atk_name}  |  Budget: {args.budget}  |  Model: {args.model}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)

            # clean baseline
            det_clean = FraudDetector(model_type=args.model,
                                      num_features=data.x.shape[1])
            det_clean.fit(data, verbose=False)
            clean_m = eval_detector(det_clean, data)
            clean_p = det_clean.predict(data.to(CFG.device))

            # attack
            kw = dict(budget_nodes=args.budget,
                      edges_per_node=CFG.attack.edges_per_node, seed=seed)
            if atk_name == "label_delay":
                kw["delay_window"] = 3
            attack = ATTACK_REGISTRY[atk_name](**kw)
            orig_n = data.num_nodes
            poisoned = attack.execute(data, target_class=1)

            # retrain
            det_p = FraudDetector(model_type=args.model,
                                  num_features=data.x.shape[1])
            det_p.fit(poisoned, verbose=False)
            poison_m = eval_detector(det_p, poisoned)
            poison_p = det_p.predict(poisoned.to(CFG.device))

            asr = attack_success_rate(clean_p, poison_p, data)
            amp = amplification_factor(clean_p, poison_p, poisoned, orig_n)
            cost = attack.gas_cost()

            r = dict(
                seed=seed,
                f1_clean=clean_m["f1"], f1_poison=poison_m["f1"],
                rec_clean=clean_m["rec"], rec_poison=poison_m["rec"],
                asr=asr["asr"], amp=amp["amp"],
                usd=cost["usd"],
                nodes_inj=cost["nodes"], edges_inj=cost["edges"],
            )
            seed_results.append(r)
            print(f"  seed={seed} | F1: {r['f1_clean']:.4f} -> {r['f1_poison']:.4f} "
                  f"| ASR={r['asr']:.3f} | ${r['usd']:.2f}")

        agg = aggregate_seeds(seed_results)
        print(f"\n  MEAN: F1 drop={agg['f1_clean_mean'] - agg['f1_poison_mean']:.4f}  "
              f"ASR={agg['asr_mean']:.3f}±{agg['asr_std']:.3f}")

        save_json(dict(attack=atk_name, budget=args.budget,
                       per_seed=seed_results, aggregated=agg),
                  f"attack_{atk_name}_b{args.budget}.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
