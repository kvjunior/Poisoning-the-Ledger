"""
Global configuration — Poisoning the Ledger v2.
Hardware-aware defaults for 4×RTX 3090 (24 GB each), 384 GB RAM.
"""
import os, torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_dir: str = "./datasets"
    elliptic_dir: str = "./datasets/elliptic"
    num_features: int = 166
    temporal_split_step: int = 34
    use_temporal_split: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    heads: int = 4
    aggr: str = "mean"


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    class_weight_illicit: float = 10.0
    # Multi-seed for statistical rigor
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])


@dataclass
class AttackConfig:
    budgets: List[int] = field(
        default_factory=lambda: [10, 25, 50, 100, 200, 500]
    )
    edges_per_node: int = 5
    # Gas economics (Ethereum mainnet, configurable)
    gas_per_tx: int = 21_000
    gas_price_gwei: float = 30.0
    eth_price_usd: float = 3500.0


@dataclass
class DefenseConfig:
    quarantine_age: int = 2
    trim_ratio: float = 0.1
    provenance_time_dim: int = 16
    provenance_age_decay: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    output_dir: str = "./outputs"
    figure_dir: str = "./figures"
    ckpt_dir: str = "./checkpoints"
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for d in (self.output_dir, self.figure_dir, self.ckpt_dir):
            os.makedirs(d, exist_ok=True)


CFG = Config()
