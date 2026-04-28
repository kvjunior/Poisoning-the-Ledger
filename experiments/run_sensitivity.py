#!/usr/bin/env python3
"""
Economic sensitivity analysis: how attack cost varies with gas/ETH price.

Usage:
    python -m experiments.run_sensitivity --results ./outputs
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from evaluation.metrics import gas_to_usd


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--results", default="./outputs")
    args = pa.parse_args()

    # Load attack results
    atk_path = os.path.join(args.results, "attack_results.json")
    if not os.path.exists(atk_path):
        print(f"Run the full pipeline first: {atk_path} not found")
        return
    with open(atk_path) as f:
        attack_data = json.load(f)

    gas_prices = [10, 20, 30, 50, 100, 200]  # gwei
    eth_prices = [1500, 2500, 3500, 5000]  # USD

    rows = []
    for r in attack_data:
        for gp in gas_prices:
            for ep in eth_prices:
                c = gas_to_usd(r.get("nodes_inj", 0), r.get("edges_inj", 0),
                               gas_gwei=gp, eth_usd=ep)
                rows.append(dict(
                    attack=r["attack"], budget=r["budget"],
                    gas_gwei=gp, eth_usd=ep, usd_cost=c["usd"],
                    f1_drop=r.get("f1_clean", 0) - r.get("f1_poison", 0),
                    asr=r.get("asr", 0),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, "sensitivity_analysis.csv"), index=False)

    # Figure: cost sensitivity heatmap for strongest attack
    best_atk = df.loc[df.groupby("attack")["asr"].idxmax().values[0], "attack"]
    best_budget = df.loc[df["asr"].idxmax(), "budget"]
    sub = df[(df["attack"] == best_atk) & (df["budget"] == best_budget)]

    pivot = sub.pivot_table(values="usd_cost", index="gas_gwei",
                            columns="eth_usd", aggfunc="first")

    fig, ax = plt.subplots(figsize=(6, 4))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("ETH Price (USD)")
    ax.set_ylabel("Gas Price (Gwei)")
    ax.set_title(f"Attack cost (USD) — {best_atk}, budget={best_budget}")

    fig_path = os.path.join(CFG.figure_dir, "fig7_sensitivity.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] {fig_path}")

    # Attacker ROI table
    print("\n  Attacker ROI analysis (avg fraud tx = $5,000)")
    avg_fraud = 5000
    for _, r in df[(df["gas_gwei"] == 30) & (df["eth_usd"] == 3500)].iterrows():
        fraud_evaded = int(r["asr"] * r["budget"])
        protected = fraud_evaded * avg_fraud
        roi = protected / max(r["usd_cost"], 0.01)
        print(f"    {r['attack']:15s} budget={int(r['budget']):4d} | "
              f"cost=${r['usd_cost']:8.2f} | protected=${protected:10.0f} | "
              f"ROI={roi:8.0f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
