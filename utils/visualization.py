"""
Generate all paper figures. Publication-quality, ACM sigconf compatible.

Fig 1: Cost vs F1 degradation (main result)
Fig 2: ASR vs budget per attack
Fig 3: Amplification heatmap
Fig 4: Defense comparison bar chart
Fig 5: Training curves
Fig 6: Structural statistics before/after
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
_C = {"camouflage": "#2166AC", "label_delay": "#B2182B", "topology": "#4DAF4A"}
_M = {"camouflage": "o", "label_delay": "s", "topology": "D"}

def _save(fig, name):
    p = os.path.join(CFG.figure_dir, name)
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def fig_cost_vs_degradation(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for at in df["attack"].unique():
        s = df[df["attack"] == at].sort_values("usd")
        ax.plot(s["usd"], s["f1_drop_pct"], marker=_M.get(at, "o"),
                color=_C.get(at, "#333"), label=at.replace("_", " ").title(),
                lw=2, ms=6)
    ax.set_xlabel("Attack cost (USD)")
    ax.set_ylabel("F1-score degradation (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    if (df["usd"] > 0).any(): ax.set_xscale("log")
    _save(fig, "fig1_cost_vs_degradation.pdf")


def fig_asr_vs_budget(results):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for at in sorted({r["attack"] for r in results}):
        sub = sorted([r for r in results if r["attack"] == at],
                     key=lambda x: x["budget"])
        ax.plot([r["budget"] for r in sub],
                [r["asr"]*100 for r in sub],
                marker=_M.get(at, "o"), color=_C.get(at, "#333"),
                label=at.replace("_", " ").title(), lw=2, ms=6)
    ax.set_xlabel("Injected nodes"); ax.set_ylabel("Attack success rate (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "fig2_asr_vs_budget.pdf")


def fig_amplification(results):
    attacks = sorted({r["attack"] for r in results})
    budgets = sorted({r["budget"] for r in results})
    mat = np.zeros((len(attacks), len(budgets)))
    for r in results:
        i = attacks.index(r["attack"])
        j = budgets.index(r["budget"])
        mat[i, j] = r.get("amp", 0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=budgets,
                yticklabels=[a.replace("_", " ").title() for a in attacks],
                cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Injected nodes"); ax.set_ylabel("Attack")
    _save(fig, "fig3_amplification.pdf")


def fig_defense_comparison(results):
    """results: {defense: {attack: {f1: ...}}}"""
    defenses = list(results.keys())
    attacks = list(next(iter(results.values())).keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(attacks))
    w = 0.8 / len(defenses)
    colors = ["#2166AC", "#B2182B", "#4DAF4A", "#984EA3", "#FF7F00"]
    for i, df_name in enumerate(defenses):
        vals = [results[df_name][a].get("f1", 0) * 100 for a in attacks]
        ax.bar(x + i * w, vals, w, label=df_name.replace("_", " ").title(),
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Attack"); ax.set_ylabel("F1-score (%)")
    ax.set_xticks(x + w * (len(defenses) - 1) / 2)
    ax.set_xticklabels([a.replace("_", " ").title() for a in attacks])
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig4_defense_comparison.pdf")


def fig_training_curves(history, name="fig5_training.pdf"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    eps = [h["epoch"] for h in history]
    a1.plot(eps, [h["train_loss"] for h in history], label="Train", lw=1.2)
    a1.plot(eps, [h["val_loss"] for h in history], label="Val", lw=1.2)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend()
    a1.grid(True, alpha=0.3)
    a2.plot(eps, [h["val_f1"] for h in history], color="#B2182B", lw=1.2)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Val F1"); a2.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, name)


def fig_structural_comparison(clean_stats, poison_stats):
    """Bar chart of structural features before/after poisoning."""
    keys = ["avg_deg_illicit", "avg_deg_licit", "illicit_ratio"]
    labels = ["Avg degree\n(illicit)", "Avg degree\n(licit)", "Illicit ratio"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, [clean_stats.get(k, 0) for k in keys], w,
           label="Clean", color="#2166AC", alpha=0.85)
    ax.bar(x + w/2, [poison_stats.get(k, 0) for k in keys], w,
           label="Poisoned", color="#B2182B", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig6_structural.pdf")
