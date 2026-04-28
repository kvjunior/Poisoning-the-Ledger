#!/usr/bin/env python3
"""
Full experimental pipeline for "Poisoning the Ledger".

Produces all results, tables, and figures for the paper:
  Phase 1: Train clean baselines (GCN, GAT, SAGE) × 5 seeds
  Phase 2: Execute 3 attacks × 6 budgets × 5 seeds
  Phase 3: Evaluate 3 defenses × 3 attacks × 5 seeds
  Phase 4: Ablation studies
  Phase 5: Economic analysis + all figures

Usage:
    # Full run on Elliptic (requires dataset)
    python -m experiments.run_full_pipeline

    # Quick test with synthetic data
    python -m experiments.run_full_pipeline --synthetic --quick

    # Single model, single seed (debugging)
    python -m experiments.run_full_pipeline --synthetic --quick --model gat --seeds 42
"""
import os, sys, time, argparse, json
import torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG
from data.loader import load_elliptic, generate_synthetic
from data.preprocessing import normalize_features, graph_stats, clone_data
from models.detector import FraudDetector
from attacks.attacks import ATTACK_REGISTRY
from defenses.defenses import (TemporalConsistencyDefense, RobustGNN,
                                ProvenanceGNN)
from evaluation.metrics import (eval_detector, attack_success_rate,
                                 amplification_factor, aggregate_seeds,
                                 cost_effectiveness_table, gas_to_usd)
from utils.logger import save_json
from utils.visualization import (fig_cost_vs_degradation, fig_asr_vs_budget,
                                  fig_amplification, fig_defense_comparison,
                                  fig_training_curves, fig_structural_comparison)


def banner(text):
    print(f"\n{'='*70}\n{text}\n{'='*70}")


# ======================================================================
#  PHASE 1: CLEAN BASELINES
# ======================================================================
def phase1_baselines(data, model_types, seeds, verbose=True):
    banner("PHASE 1: CLEAN BASELINES")
    results = {}
    for mt in model_types:
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if verbose:
                print(f"\n  {mt.upper()} seed={seed}")
            det = FraudDetector(model_type=mt, num_features=data.x.shape[1])
            det.fit(data, verbose=verbose)
            m = eval_detector(det, data)
            m["seed"] = seed
            seed_results.append(m)

        agg = aggregate_seeds(seed_results)
        results[mt] = dict(per_seed=seed_results, aggregated=agg)
        print(f"\n  {mt.upper()} | F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}  "
              f"P={agg['prec_mean']:.4f}  R={agg['rec_mean']:.4f}")

    return results


# ======================================================================
#  PHASE 2: ATTACKS
# ======================================================================
def phase2_attacks(data, model_type, budgets, seeds, verbose=True):
    banner(f"PHASE 2: ATTACKS (detector={model_type.upper()})")

    # train one clean detector per seed for comparison
    clean_preds = {}
    clean_metrics = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        det = FraudDetector(model_type=model_type,
                            num_features=data.x.shape[1])
        det.fit(data, verbose=False)
        clean_preds[seed] = det.predict(data.to(CFG.device))
        clean_metrics[seed] = eval_detector(det, data)

    all_results = []

    for atk_name, atk_cls in ATTACK_REGISTRY.items():
        print(f"\n--- Attack: {atk_name} ---")
        for budget in budgets:
            seed_results = []
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)

                # construct attack
                kw = dict(budget_nodes=budget,
                          edges_per_node=CFG.attack.edges_per_node, seed=seed)
                if atk_name == "label_delay":
                    kw["delay_window"] = 3
                attack = atk_cls(**kw)

                orig_n = data.num_nodes
                poisoned = attack.execute(data, target_class=1)

                # retrain on poisoned data
                p_det = FraudDetector(model_type=model_type,
                                     num_features=data.x.shape[1])
                p_det.fit(poisoned, verbose=False)

                p_metrics = eval_detector(p_det, poisoned)
                p_pred = p_det.predict(poisoned.to(CFG.device))

                asr = attack_success_rate(
                    clean_preds[seed], p_pred, data)
                amp = amplification_factor(
                    clean_preds[seed], p_pred, poisoned, orig_n)
                cost = attack.gas_cost()

                r = dict(
                    attack=atk_name, budget=budget, seed=seed,
                    nodes_inj=cost["nodes"], edges_inj=cost["edges"],
                    usd=cost["usd"],
                    f1_clean=clean_metrics[seed]["f1"],
                    f1_poison=p_metrics["f1"],
                    rec_clean=clean_metrics[seed]["rec"],
                    rec_poison=p_metrics["rec"],
                    asr=asr["asr"], amp=amp["amp"],
                )
                seed_results.append(r)

            # aggregate across seeds for this attack+budget
            agg = aggregate_seeds(seed_results)
            mean_r = dict(
                attack=atk_name, budget=budget,
                nodes_inj=seed_results[0]["nodes_inj"],
                edges_inj=seed_results[0]["edges_inj"],
                usd=seed_results[0]["usd"],
                f1_clean=agg["f1_clean_mean"],
                f1_poison=agg["f1_poison_mean"],
                f1_drop=agg["f1_clean_mean"] - agg["f1_poison_mean"],
                asr=agg["asr_mean"], asr_std=agg["asr_std"],
                amp=agg["amp_mean"],
            )
            all_results.append(mean_r)

            print(f"  budget={budget:4d} | F1: {mean_r['f1_clean']:.4f} -> "
                  f"{mean_r['f1_poison']:.4f} "
                  f"(-{mean_r['f1_drop']:.4f}) | ASR={mean_r['asr']:.3f} | "
                  f"${mean_r['usd']:.2f}")

    return all_results


# ======================================================================
#  PHASE 3: DEFENSES
# ======================================================================
def phase3_defenses(data, model_type, budgets, seeds, verbose=True):
    banner(f"PHASE 3: DEFENSES (detector={model_type.upper()})")

    max_budget = max(budgets)
    defense_results = {}

    for atk_name, atk_cls in ATTACK_REGISTRY.items():
        print(f"\n--- Defending against: {atk_name} (budget={max_budget}) ---")
        defense_results[atk_name] = {}

        for def_name in ["no_defense", "temporal", "robust", "provenance"]:
            seed_results = []
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)

                # reproduce attack
                kw = dict(budget_nodes=max_budget,
                          edges_per_node=CFG.attack.edges_per_node, seed=seed)
                if atk_name == "label_delay":
                    kw["delay_window"] = 3
                poisoned = atk_cls(**kw).execute(data, target_class=1)

                # apply defense + train
                if def_name == "no_defense":
                    det = FraudDetector(model_type=model_type,
                                       num_features=data.x.shape[1])
                    det.fit(poisoned, verbose=False)
                    m = eval_detector(det, poisoned)

                elif def_name == "temporal":
                    defended = TemporalConsistencyDefense().apply(poisoned)
                    det = FraudDetector(model_type=model_type,
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

                seed_results.append(m)

            agg = aggregate_seeds(seed_results)
            defense_results[atk_name][def_name] = {
                "f1": agg["f1_mean"], "f1_std": agg["f1_std"],
                "prec": agg["prec_mean"], "rec": agg["rec_mean"],
            }
            print(f"  {def_name:15s} | F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}")

    return defense_results


# ======================================================================
#  PHASE 4: ABLATIONS
# ======================================================================
def phase4_ablations(data, model_type, seed=42):
    banner("PHASE 4: ABLATIONS")
    torch.manual_seed(seed); np.random.seed(seed)

    budget = 100
    results = {}

    # Ablation A: feature strategy for camouflage
    print("\n  Ablation A: Feature generation strategy")
    for strategy in ["mimic_licit", "random"]:
        from attacks.attacks import NeighborhoodCamouflage
        atk = NeighborhoodCamouflage(budget_nodes=budget,
                                      edges_per_node=5, seed=seed)
        # override feature generation for random
        if strategy == "random":
            orig_fn = atk._sample_licit_features
            atk._sample_licit_features = lambda d, k: torch.randn(k, d.x.shape[1]) * 0.1

        poisoned = atk.execute(data, target_class=1)
        det = FraudDetector(model_type=model_type, num_features=data.x.shape[1])
        det.fit(poisoned, verbose=False)
        m = eval_detector(det, poisoned)
        results[f"feat_{strategy}"] = m
        print(f"    {strategy:15s} | F1={m['f1']:.4f}")

    # Ablation B: edges per injected node
    print("\n  Ablation B: Edges per injected node")
    for epn in [1, 3, 5, 10]:
        atk = NeighborhoodCamouflage(budget_nodes=budget,
                                      edges_per_node=epn, seed=seed)
        poisoned = atk.execute(data, target_class=1)
        det = FraudDetector(model_type=model_type, num_features=data.x.shape[1])
        det.fit(poisoned, verbose=False)
        m = eval_detector(det, poisoned)
        results[f"epn_{epn}"] = m
        print(f"    edges_per_node={epn:2d} | F1={m['f1']:.4f}")

    # Ablation C: model architecture sensitivity
    print("\n  Ablation C: Architecture sensitivity")
    for mt in ["gcn", "gat", "sage"]:
        atk = NeighborhoodCamouflage(budget_nodes=budget,
                                      edges_per_node=5, seed=seed)
        poisoned = atk.execute(data, target_class=1)
        det = FraudDetector(model_type=mt, num_features=data.x.shape[1])
        det.fit(poisoned, verbose=False)
        m = eval_detector(det, poisoned)
        results[f"arch_{mt}"] = m
        print(f"    {mt:5s} | F1={m['f1']:.4f}")

    return results


# ======================================================================
#  MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", default="gat", choices=["gcn","gat","sage","all"])
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds, e.g. '42,123'")
    parser.add_argument("--output", default="./outputs")
    args = parser.parse_args()

    CFG.output_dir = args.output
    CFG.figure_dir = os.path.join(args.output, "figures")
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.figure_dir, exist_ok=True)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.quick:
        seeds = [42]
    else:
        seeds = CFG.train.seeds

    budgets = [10, 50] if args.quick else CFG.attack.budgets
    model_types = ["gcn", "gat", "sage"] if args.model == "all" else [args.model]
    primary = model_types[0]

    t0 = time.time()

    # ---- Load data ----
    banner("LOADING DATA")
    if args.synthetic:
        data = generate_synthetic(n=50000 if not args.quick else 10000,
                                  m=200000 if not args.quick else 40000)
    else:
        data = load_elliptic()
    data = normalize_features(data)
    stats = graph_stats(data)
    save_json(stats, "graph_stats.json")

    # ---- Phase 1 ----
    baseline = phase1_baselines(data, model_types, seeds, verbose=not args.quick)
    save_json({mt: r["aggregated"] for mt, r in baseline.items()},
              "baselines.json")

    # save training curves for primary model
    if baseline[primary]["per_seed"]:
        det_for_curves = FraudDetector(model_type=primary,
                                       num_features=data.x.shape[1])
        torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
        det_for_curves.fit(data, verbose=False)
        if det_for_curves.history:
            fig_training_curves(det_for_curves.history)

    # ---- Phase 2 ----
    attack_results = phase2_attacks(data, primary, budgets, seeds,
                                    verbose=not args.quick)
    save_json(attack_results, "attack_results.json")

    # figures
    df = cost_effectiveness_table(attack_results)
    df.to_csv(os.path.join(CFG.output_dir, "cost_effectiveness.csv"), index=False)
    fig_cost_vs_degradation(df)
    fig_asr_vs_budget(attack_results)
    fig_amplification(attack_results)

    # structural comparison
    if attack_results:
        from attacks.attacks import NeighborhoodCamouflage
        atk = NeighborhoodCamouflage(budget_nodes=max(budgets), seed=seeds[0])
        poisoned_for_stats = atk.execute(data, target_class=1)
        p_stats = graph_stats(poisoned_for_stats)
        fig_structural_comparison(stats, p_stats)

    # ---- Phase 3 ----
    defense_results = phase3_defenses(data, primary, budgets, seeds,
                                       verbose=not args.quick)
    save_json(defense_results, "defense_results.json")
    fig_defense_comparison(defense_results)

    # ---- Phase 4 ----
    ablation = phase4_ablations(data, primary, seed=seeds[0])
    save_json(ablation, "ablation_results.json")

    # ---- Summary ----
    elapsed = time.time() - t0
    banner("COMPLETE")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Results: {CFG.output_dir}/")
    print(f"Figures: {CFG.figure_dir}/")
    print(f"\nKey outputs:")
    print(f"  baselines.json          — Table 2 (clean detector performance)")
    print(f"  attack_results.json     — Table 3 (attack effectiveness)")
    print(f"  cost_effectiveness.csv  — Table 4 (economics)")
    print(f"  defense_results.json    — Table 5 (defense comparison)")
    print(f"  ablation_results.json   — Table 6 (ablation study)")
    print(f"  fig1_cost_vs_degradation.pdf  — Main result figure")
    print(f"  fig2_asr_vs_budget.pdf")
    print(f"  fig3_amplification.pdf")
    print(f"  fig4_defense_comparison.pdf")
    print(f"  fig5_training.pdf")
    print(f"  fig6_structural.pdf")


if __name__ == "__main__":
    main()
