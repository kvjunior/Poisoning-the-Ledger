# Poisoning the Ledger

Code, defenses, and evaluation pipeline for the paper:

> **Poisoning the Ledger: Economic Append-Only Data Poisoning Attacks Against Graph-Based Blockchain Fraud Detection**
> Submission under double-blind review, CCS 2026.

This repository implements three append-only poisoning attacks against
graph-based blockchain fraud detectors and three mechanism-matched defenses,
together with two published-defense baselines and the full evaluation pipeline
used to produce every numerical result and figure in the paper.

The repository is anonymized for double-blind review. No commit history,
author identifiers, or institutional information is included. Artifacts will
be de-anonymized at camera-ready.

---

## What this artifact contains

The repository implements the contributions described in Sections 4 (attacks),
5 (defenses), and 6 (evaluation) of the paper, plus the additional analyses
in Appendix E.

| Component | Section | Files |
| --- | --- | --- |
| Neighborhood Camouflage attack | §4.2 (Algorithm 1) | `attacks/camouflage.py` |
| Label-Delay Exploitation attack | §4.3 (Algorithm 2) | `attacks/label_delay.py` |
| Topology Manipulation attack | §4.4 (Algorithm 3) | `attacks/topology.py` |
| Temporal Consistency Filtering | §5.2 | `defenses/temporal_consistency.py` |
| Robust Aggregation (trimmed mean) | §5.3 (Eq. 5) | `defenses/robust_aggregation.py` |
| Provenance-Aware GNN | §5.4 (Eq. 6) | `defenses/provenance_aware.py` |
| RGCN baseline | §5.5 | `defenses/rgcn_baseline.py` |
| Gradient-shaping DP-SGD baseline | §5.5 | `defenses/dpsgd_baseline.py` |
| GAT, GCN, GraphSAGE detectors | §6.1 | `models/` |
| Elliptic preprocessing pipeline | §6.1 | `data/preprocess.py` |
| Attack–budget grid evaluation | §6.2, Table 4 | `experiments/run_attacks.py` |
| Defense-effectiveness evaluation | §6.3, Table 5 | `experiments/run_defenses.py` |
| Cross-architecture transferability | §6.4, Table 6 | `experiments/run_transferability.py` |
| Economic analysis & gas-price sweep | §6.5, Table 7, Fig. 5 | `experiments/run_economics.py` |
| Ablations | §6.6, Table 8 | `experiments/run_ablations.py` |
| Per-seed and structural analyses | App. E | `experiments/run_appendix_e.py` |
| All paper figures (Figs. 1–5, App. E) | — | `evaluation/figures/` |

---

## Repository layout

```
.
├── attacks/             # the three poisoning attacks
│   ├── camouflage.py
│   ├── label_delay.py
│   ├── topology.py
│   └── __init__.py
├── data/                # Elliptic preprocessing and cached splits
│   ├── preprocess.py
│   ├── splits.py
│   └── README.md
├── defenses/            # three defenses and two published baselines
│   ├── temporal_consistency.py
│   ├── robust_aggregation.py
│   ├── provenance_aware.py
│   ├── rgcn_baseline.py
│   ├── dpsgd_baseline.py
│   └── __init__.py
├── models/              # GNN backbones
│   ├── gat.py
│   ├── gcn.py
│   ├── graphsage.py
│   └── __init__.py
├── experiments/         # end-to-end pipelines that produce paper tables
│   ├── run_attacks.py
│   ├── run_defenses.py
│   ├── run_transferability.py
│   ├── run_economics.py
│   ├── run_ablations.py
│   └── run_appendix_e.py
├── evaluation/          # metrics, plotting, log aggregation
│   ├── metrics.py
│   ├── aggregate_logs.py
│   └── figures/
│       ├── fig1_threat_model.py
│       ├── fig2_attack_schematic.py
│       ├── fig3_cost_frontier.py
│       ├── fig4_defense_dynamics.py
│       ├── fig5_economic_landscape.py
│       └── appE_training_dynamics.py
├── utils/               # shared helpers (seeds, logging, gas calculator)
│   ├── seeds.py
│   ├── logging.py
│   └── gas.py
├── config.py            # central configuration (seeds, paths, hyperparameters)
├── reproduce.sh         # full end-to-end reproduction pipeline (~18h)
├── smoke_test.sh        # minimum-budget verification (~10min)
├── requirements         # pip dependency list
└── README.md            # this file
```

---

## System requirements

Tested on the following configuration; other configurations should work but
are not validated.

- Linux (Ubuntu 22.04 LTS)
- Python 3.10
- CUDA 12.1 with cuDNN 8.9
- 4× NVIDIA RTX 3090 (24 GB each); the full pipeline fits on a single GPU
  with reduced batch size, see `config.py`
- 384 GB system RAM (32 GB minimum for single-seed runs)
- ~50 GB free disk for cached data and intermediate logs

---

## Installation

```bash
# clone the anonymous repository
git clone <anonymous-repository-url> poisoning-the-ledger
cd poisoning-the-ledger

# create a fresh virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements
```

Dependencies pinned in `requirements`:

```
torch==2.1.0
torch-geometric==2.4.0
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==1.26.0
pandas==2.1.1
scikit-learn==1.3.1
matplotlib==3.8.0
networkx==3.1
tqdm==4.66.1
pyyaml==6.0.1
opacus==1.4.0
```

`opacus` is required only by the gradient-shaping DP-SGD baseline; the
core attack and defense pipelines do not need it.

---

## Dataset

We use the Elliptic Bitcoin transaction dataset, which is publicly available
on Kaggle under a permissive license. Place the three raw CSVs in
`data/elliptic_raw/`:

```
data/elliptic_raw/elliptic_txs_features.csv
data/elliptic_raw/elliptic_txs_edgelist.csv
data/elliptic_raw/elliptic_txs_classes.csv
```

Then run the preprocessing pipeline once:

```bash
python -m data.preprocess
```

This produces:

- `data/cache/features.npy` — node feature matrix, 203,769 × 166
- `data/cache/edges.npy` — directed edge list, 234,355 × 2
- `data/cache/labels.npy` — labels in {0, 1, ⊥} per Section 3.1
- `data/cache/timesteps.npy` — per-node timestep, in {1, …, 49}
- `data/cache/splits.npz` — temporal 80/20 split on timesteps 1–34 plus
  the held-out test set of timesteps 35–49 (Section 6.1)

Cached files total ~1.2 GB. The raw CSVs are not redistributed in this
repository; download them from Kaggle.

---

## Quick start: smoke test

To verify a working installation without committing to the full 18-hour
pipeline, run:

```bash
bash smoke_test.sh
```

This executes a single seed of Neighborhood Camouflage at the smallest
budget (N = 10) against the GAT primary detector. It completes in
approximately 10 minutes on a single RTX 3090 and prints the post-attack
F1 to stdout. The expected value is 85.12 ± 0.5 F1 (matching the
budget-10 row of Table 4).

---

## Reproducing the paper

### Full pipeline

The end-to-end pipeline reproduces every numerical result in Section 6
and Appendix E:

```bash
bash reproduce.sh
```

This runs:

1. Five seeds × six budgets × three attacks against the GAT detector
   (Table 4, Figure 3)
2. Five seeds × five defenses × three attacks at budget 200
   (Table 5, Figure 4)
3. Cross-architecture transferability on Camouflage (Table 6)
4. Economic sensitivity grid: 6 gas prices × 4 ETH prices (Table 7, Figure 5)
5. Four ablation panels (Table 8)
6. Appendix-E experiments: per-seed breakdown, structural statistics,
   extended τ ablation, training dynamics, cross-budget defense comparison

Wall-clock time is approximately 18 hours on 4× RTX 3090.

### Reduced-budget reproductions

For reviewers with limited compute, three smaller reproduction targets are
supported via `reproduce.sh` flags:

```bash
# single seed, three attacks, three budgets, no transferability or ablations
# (~4 hours)
bash reproduce.sh --single-seed

# Camouflage only, all five seeds, all six budgets
# (~90 minutes)
bash reproduce.sh --camouflage-only

# regenerate figures only, from cached metric logs in evaluation/cache/
# (~2 minutes)
bash reproduce.sh --figures-only
```

### Per-experiment commands

Each experiment can be run individually:

```bash
# Table 4 / Figure 3
python -m experiments.run_attacks --seeds 42,123,456,789,1024 \
    --budgets 10,25,50,100,200,500 --architecture gat

# Table 5 / Figure 4
python -m experiments.run_defenses --budget 200 \
    --defenses temporal_consistency,robust_aggregation,provenance_aware,rgcn,dpsgd

# Table 6
python -m experiments.run_transferability --budget 200 --attack camouflage

# Table 7 / Figure 5
python -m experiments.run_economics

# Table 8
python -m experiments.run_ablations

# Appendix E (per-seed, structural, τ-extended, training-dynamics, cross-budget)
python -m experiments.run_appendix_e
```

---

## Configuration

All hyperparameters and paths are centralized in `config.py`. Notable values:

| Parameter | Value | Reference |
| --- | --- | --- |
| Seeds | {42, 123, 456, 789, 1024} | §6.1 |
| Primary detector | 3-layer GAT, 4 heads, hidden dim 128 | §6.1 |
| Camouflage feature noise η | 0.08 | §4.2 |
| Label-Delay mixing weight α | 0.6 | Eq. 4 |
| Label-Delay noise scale β | 0.25 | §4.3 |
| Topology cluster probability p_c | 0.15 | §4.4 |
| Reference gas price | 30 gwei | §3.2 |
| Reference ETH/USD rate | 3,500 | §3.2 |
| Mean fraud-transaction value | $5,000 | §6.5 |
| Time encoding dim d_time | 16 | Eq. 6 |
| Trim ratio ρ | 0.1 | §5.3 |
| Age threshold A | 2 timesteps | §5.2 |
| DP-SGD clipping C | 1.0 | §5.5 |
| DP-SGD noise σ | 0.5 | §5.5 |

Override any value via the command line or by editing `config.py` directly.

---

## Output organization

All experiments write to `evaluation/cache/` and `evaluation/figures/`:

```
evaluation/cache/
├── attacks/             # raw per-seed metric logs (JSON)
├── defenses/
├── transferability/
├── economics/
├── ablations/
└── appendix_e/

evaluation/figures/
├── fig1_threat_model.{pdf,png}
├── fig2_attack_schematic.{pdf,png}
├── fig3_cost_frontier.{pdf,png}
├── fig4_defense_dynamics.{pdf,png}
├── fig5_economic_landscape.{pdf,png}
└── appE_training_dynamics.{pdf,png}
```

Tables in the paper are produced from the JSON logs by:

```bash
python -m evaluation.aggregate_logs --table 4   # any of {3,4,5,6,7,8}
```

This prints the LaTeX table body to stdout.

---

## Reproducing individual figures

Each figure has a dedicated rendering script that reads from
`evaluation/cache/` and writes to `evaluation/figures/`:

```bash
python -m evaluation.figures.fig3_cost_frontier
python -m evaluation.figures.fig4_defense_dynamics
python -m evaluation.figures.fig5_economic_landscape
python -m evaluation.figures.appE_training_dynamics
```

Figures 1 (threat model schematic) and 2 (attack schematic) are conceptual
diagrams; they do not depend on experimental data and are produced by
matplotlib scripts in `evaluation/figures/`.

---

## Validation

The repository ships with frozen reference logs in `evaluation/cache/`
covering all five seeds. After running `reproduce.sh`, compare your
output against the reference:

```bash
python -m evaluation.aggregate_logs --validate
```

This prints a per-table diff. All values should match within numerical
floating-point tolerance.

---

## Limitations

This artifact reproduces the paper's main claims but inherits the paper's
scope:

- Single dataset (Elliptic). Cross-chain evaluation is left to follow-up
  work; see Section 7.2.
- Adaptive-attack experiments against the Provenance-Aware GNN are
  preliminary; see Section 7.1.
- The fraud-transaction-value assumption used in the ROI calculations
  (Section 6.5) is generic; alternative values rescale the ROI column
  linearly.

---

## Ethical use

Section 7.4 and Appendix B of the paper describe the dual-use considerations
of this work and our responsible-disclosure process with commercial tracing
providers. The implementations released here are calibrated for defensive
research use, not for turn-key offensive deployment against any production
system. All experiments use the public Elliptic dataset; no real cryptocurrency
addresses were created and no transactions were broadcast.

---

## Anonymous review notes

The repository is hosted at anonymous.4open.science. Commit history,
author metadata, and institutional identifiers have been stripped. The
de-anonymized repository will be released after the review period, with
the same code organization and hyperparameter values.
