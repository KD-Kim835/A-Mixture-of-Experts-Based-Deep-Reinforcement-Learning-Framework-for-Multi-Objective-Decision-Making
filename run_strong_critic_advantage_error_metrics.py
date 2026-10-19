from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, TrainConfig, ensure_output_dirs
from strong_critic_simulation import StrongCriticConfig, run_one

MODELS = ("FC", "MH", "MoE", "MH+MoE")


SUMMARY_PATH = RESULTS_DIR / "strong_critic_regime_simulation_30seed_advantage_error_summary.csv"
AGG_PATH = RESULTS_DIR / "strong_critic_regime_simulation_30seed_advantage_error_aggregated.csv"
TEST_PATH = RESULTS_DIR / "strong_critic_regime_simulation_30seed_advantage_error_paired_tests.csv"
SEEDS_30 = tuple(11 * i for i in range(1, 31))

METRICS = [
    "test_mse",
    "advantage_error_mae",
    "gradient_norm_mean",
    "gate_regime_alignment",
]
SUMMARY_COLUMNS = ["model", "seed", "heterogeneity_strength", *METRICS]


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys() -> set[tuple[float, int, str]]:
    if not SUMMARY_PATH.exists():
        return set()
    df = pd.read_csv(SUMMARY_PATH)
    return {(float(row.heterogeneity_strength), int(row.seed), str(row.model)) for row in df.itertuples(index=False)}


def signflip_p(diff: np.ndarray, n_mc: int = 200_000, seed: int = 3701) -> float:
    vals = np.asarray(diff, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan
    obs = abs(float(vals.mean()))
    if vals.size <= 20:
        sims = [abs(float((vals * np.asarray(signs)).mean())) for signs in itertools.product([-1, 1], repeat=vals.size)]
        return float(sum(v >= obs - 1e-12 for v in sims) / len(sims))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, vals.size))
    sims = np.abs((signs * vals).mean(axis=1))
    return float((np.sum(sims >= obs - 1e-12) + 1) / (n_mc + 1))


def write_tables(summary: pd.DataFrame) -> None:
    agg = summary.groupby(["heterogeneity_strength", "model"])[METRICS].agg(["mean", "std", "sem"]).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg.to_csv(AGG_PATH, index=False)

    paired_metrics = [("test_mse", "lower"), ("advantage_error_mae", "lower"), ("gradient_norm_mean", "lower")]
    rows = []
    row_seed = 0
    for h, sub in summary.groupby("heterogeneity_strength"):
        for model_a in ("MoE", "MH+MoE"):
            for model_b in ("FC", "MH"):
                for metric, direction in paired_metrics:
                    pa = sub[sub["model"] == model_a].set_index("seed")[metric]
                    pb = sub[sub["model"] == model_b].set_index("seed")[metric]
                    seeds = pa.index.intersection(pb.index)
                    diff = pa.loc[seeds].to_numpy() - pb.loc[seeds].to_numpy()
                    rows.append(
                        {
                            "heterogeneity_strength": h,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric,
                            "direction": direction,
                            "n_pairs": len(seeds),
                            "mean_a": float(pa.loc[seeds].mean()),
                            "mean_b": float(pb.loc[seeds].mean()),
                            "diff_a_minus_b": float(diff.mean()),
                            "signflip_p_two_sided": signflip_p(diff, seed=3701 + row_seed),
                        }
                    )
                    row_seed += 1
    pd.DataFrame(rows).to_csv(TEST_PATH, index=False)


def main() -> None:
    ensure_output_dirs()
    cfg = StrongCriticConfig(seeds=SEEDS_30)
    train_cfg = TrainConfig(epochs=85, batch_size=192, gate_entropy_weight=0.001, load_balance_weight=0.003)
    done = completed_keys()
    total = len(cfg.heterogeneity_levels) * len(cfg.seeds) * len(MODELS)
    count = len(done)
    for h in cfg.heterogeneity_levels:
        for seed in cfg.seeds:
            for model in MODELS:
                key = (float(h), int(seed), model)
                if key in done:
                    continue
                row = {key: value for key, value in run_one(model, seed, h, cfg, train_cfg).items() if key in SUMMARY_COLUMNS}
                append_csv(SUMMARY_PATH, [row])
                done.add(key)
                count += 1
                print(f"[{count}/{total}] h={h:g} seed={seed} model={model}", flush=True)
    summary = pd.read_csv(SUMMARY_PATH)[SUMMARY_COLUMNS]
    summary.to_csv(SUMMARY_PATH, index=False)
    write_tables(summary)
    print(f"Wrote 30-seed strong critic advantage-error metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

