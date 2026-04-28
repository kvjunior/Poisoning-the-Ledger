#!/usr/bin/env python3
"""
Generate LaTeX tables for the paper from experiment JSON files.

Usage:
    python -m utils.tables --results ./outputs

Produces:
  table2_baselines.tex      — Clean detector performance
  table3_attacks.tex         — Attack effectiveness
  table4_economics.tex       — Cost-effectiveness
  table5_defenses.tex        — Defense comparison
  table6_ablations.tex       — Ablation study
"""
import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG


def load(name, results_dir):
    p = os.path.join(results_dir, name)
    if not os.path.exists(p):
        print(f"  [skip] {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


def pm(mean_key, std_key, d, pct=False):
    """Format mean ± std."""
    m = d.get(mean_key, 0)
    s = d.get(std_key, 0)
    if pct:
        return f"{m*100:.2f}$\\pm${s*100:.2f}"
    return f"{m:.4f}$\\pm${s:.4f}"


def table2_baselines(data, out_dir):
    """Table 2: Clean baseline performance across GNN architectures."""
    if not data:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Clean detection performance across GNN architectures (mean $\pm$ std over 5 seeds).}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & F1 (\%) & Precision (\%) & Recall (\%) & AUC (\%) \\",
        r"\midrule",
    ]
    for model in ["gcn", "gat", "sage"]:
        if model not in data:
            continue
        d = data[model]
        lines.append(
            f"{model.upper()} & {pm('f1_mean','f1_std',d,True)} & "
            f"{pm('prec_mean','prec_std',d,True)} & "
            f"{pm('rec_mean','rec_std',d,True)} & "
            f"{pm('auc_mean','auc_std',d,True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(lines, out_dir, "table2_baselines.tex")


def table3_attacks(data, out_dir):
    """Table 3: Attack effectiveness across budgets."""
    if not data:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attack effectiveness: F1-score degradation and attack success rate across injection budgets.}",
        r"\label{tab:attacks}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Attack & Budget & F1 Clean & F1 Poisoned & $\Delta$F1 & ASR & Cost (USD) \\",
        r"\midrule",
    ]
    for r in data:
        drop = r.get("f1_clean", 0) - r.get("f1_poison", 0)
        lines.append(
            f"{r['attack'].replace('_',' ').title()} & {r['budget']} & "
            f"{r.get('f1_clean',0):.4f} & {r.get('f1_poison',0):.4f} & "
            f"{drop:.4f} & {r.get('asr',0):.3f} & "
            f"\\${r.get('usd',0):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write(lines, out_dir, "table3_attacks.tex")


def table5_defenses(data, out_dir):
    """Table 5: Defense effectiveness."""
    if not data:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Defense effectiveness: F1-score under strongest attack with and without defenses.}",
        r"\label{tab:defenses}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Attack & Defense & F1 (\%) & $\pm$std \\",
        r"\midrule",
    ]
    for atk in data:
        for df in data[atk]:
            d = data[atk][df]
            f1 = d.get("f1", 0) * 100
            std = d.get("f1_std", 0) * 100
            lines.append(
                f"{atk.replace('_',' ').title()} & "
                f"{df.replace('_',' ').title()} & "
                f"{f1:.2f} & {std:.2f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    _write(lines, out_dir, "table5_defenses.tex")


def _write(lines, out_dir, name):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [wrote] {path}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--results", default="./outputs")
    pa.add_argument("--out", default="./outputs/tables")
    args = pa.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Generating LaTeX tables...")
    table2_baselines(load("baselines.json", args.results), args.out)
    table3_attacks(load("attack_results.json", args.results), args.out)
    table5_defenses(load("defense_results.json", args.results), args.out)
    print("Done.")


if __name__ == "__main__":
    main()
