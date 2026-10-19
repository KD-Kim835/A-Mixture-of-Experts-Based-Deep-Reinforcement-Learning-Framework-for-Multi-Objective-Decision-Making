"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / 'figures'
RESULTS_DIR = ROOT / 'results'

def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainConfig:
    epochs: int = 90
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-05
    hidden_dim: int = 96
    n_experts: int = 3
    gate_entropy_weight: float = 0.0
    load_balance_weight: float = 0.0
    device: str = 'auto'
