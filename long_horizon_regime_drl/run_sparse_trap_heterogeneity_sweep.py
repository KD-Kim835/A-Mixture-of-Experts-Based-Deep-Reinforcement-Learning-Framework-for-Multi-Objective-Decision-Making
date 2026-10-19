from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from long_horizon_regime_drl.run_advantage_sparse_trap import (  # noqa: E402
    COLORS,
    MODELS,
    AdvantageTrapConfig,
    add_random_average_metrics,
    train_one,
)


OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUMMARY_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_summary.csv"
CURVES_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_curves.csv"
AGG_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_aggregated.csv"
TEST_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_paired_tests.csv"
BOXPLOT_PATH = FIG_DIR / "sparse_trap_heterogeneity_sweep_30seed_boxplots.png"

SEEDS_30 = tuple(11 * i for i in range(1, 31))
HETEROGENEITY_LEVELS = (1.0, 2.0, 3.0, 4.0, 5.0)
MODEL_LABELS = {
    "FC": "FC",
    "MH": "MH",
    "ActorMoE": "ActorMoE",
    "MoE": "MoE",
    "MH+MoE": "MH+MoE",
}
PROPOSED_MODELS = ("MoE", "MH+MoE")
BENCHMARK_MODELS = ("FC", "MH", "ActorMoE")
BOXPLOT_MODELS = MODELS
BOXPLOT_FONT_SIZE = 30

def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def variant_name(h: float) -> str:
    return f"sparse_trap_h_sweep_30seed_h{h:g}".replace(".", "p")


def seed_from_existing_10seed() -> None:
    old_summary = OUT_DIR / "sparse_trap_heterogeneity_sweep_summary.csv"
    old_curves = OUT_DIR / "sparse_trap_heterogeneity_sweep_curves.csv"
    if not SUMMARY_PATH.exists() and old_summary.exists():
        df = pd.read_csv(old_summary)
        df = df[df["seed"].isin(SEEDS_30)].copy()
        if not df.empty:
            df.to_csv(SUMMARY_PATH, index=False)
    if not CURVES_PATH.exists() and old_curves.exists():
        df = pd.read_csv(old_curves)
        df = df[df["seed"].isin(SEEDS_30)].copy()
        if not df.empty:
            df.to_csv(CURVES_PATH, index=False)


def completed_keys() -> set[tuple[float, str, int]]:
    if not SUMMARY_PATH.exists():
        return set()
    df = pd.read_csv(SUMMARY_PATH)
    return {(float(row.heterogeneity), str(row.model), int(row.seed)) for row in df.itertuples(index=False)}


def signflip_p(diff: pd.Series, n_mc: int = 200_000, seed: int = 2701) -> float:
    vals = diff.dropna().to_numpy(dtype=float)
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


def write_summaries() -> pd.DataFrame:
    df = add_random_average_metrics(pd.read_csv(SUMMARY_PATH))
    metric_candidates = ["random_avg_return_mean", "random_avg_advantage_error_mae", "random_avg_td_advantage_error_mae", "actor_policy_gradient_norm_mean"]
    metrics = [metric for metric in metric_candidates if metric in df.columns]
    agg = df.groupby(["heterogeneity", "model"])[metrics].agg(["mean", "std", "sem"]).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg["model_label"] = agg["model"].map(MODEL_LABELS)
    agg.to_csv(AGG_PATH, index=False)

    rows = []
    row_seed = 0
    paired_metric_candidates = (("random_avg_return_mean", "higher"), ("random_avg_advantage_error_mae", "lower"), ("actor_policy_gradient_norm_mean", "lower"))
    paired_metrics = [(metric, direction) for metric, direction in paired_metric_candidates if metric in df.columns]
    for h, sub in df.groupby("heterogeneity"):
        for model_a in PROPOSED_MODELS:
            for model_b in BENCHMARK_MODELS:
                for metric, direction in paired_metrics:
                    pa = sub[sub["model"] == model_a].set_index("seed")[metric]
                    pb = sub[sub["model"] == model_b].set_index("seed")[metric]
                    seeds = pa.index.intersection(pb.index)
                    diff = pa.loc[seeds] - pb.loc[seeds]
                    rows.append(
                        {
                            "heterogeneity": h,
                            "model_a": model_a,
                            "model_a_label": MODEL_LABELS[model_a],
                            "model_b": model_b,
                            "model_b_label": MODEL_LABELS[model_b],
                            "metric": metric,
                            "direction": direction,
                            "n_pairs": len(seeds),
                            "mean_a": pa.loc[seeds].mean(),
                            "mean_b": pb.loc[seeds].mean(),
                            "diff_a_minus_b": diff.mean(),
                            "signflip_p_two_sided": signflip_p(diff, seed=2701 + row_seed),
                        }
                    )
                    row_seed += 1
    pd.DataFrame(rows).to_csv(TEST_PATH, index=False)
    keep = ["model", "seed", "heterogeneity", "parameter_count", *metrics]
    keep = [column for column in keep if column in df.columns]
    df = df[keep].copy()
    df.to_csv(SUMMARY_PATH, index=False)
    return df


def plot_metric_boxplots(df: pd.DataFrame) -> None:
    metric_candidates = [
        ("random_avg_return_mean", "DRL reward"),
        ("random_avg_td_advantage_error_mae", "Advantage error MAE"),
        ("actor_policy_gradient_norm_mean", "Actor gradient norm mean"),
    ]
    metrics = [(metric, ylabel) for metric, ylabel in metric_candidates if metric in df.columns]
    levels = sorted(df["heterogeneity"].unique())
    centers = np.arange(len(levels), dtype=float) * 1.5
    offsets = dict(zip(BOXPLOT_MODELS, np.linspace(-0.4, 0.4, len(BOXPLOT_MODELS))))
    ncols = 3
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(36, 10), sharex=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for model_name in BOXPLOT_MODELS:
            values = [
                df[(df["model"] == model_name) & (df["heterogeneity"] == h)][metric].dropna().to_numpy()
                for h in levels
            ]
            positions = [center + offsets[model_name] for center in centers]
            color = COLORS[model_name]
            ax.boxplot(
                values,
                positions=positions,
                widths=0.2,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.45},
                whiskerprops={"color": color, "linewidth": 0.9},
                capprops={"color": color, "linewidth": 0.9},
            )
            for pos, seed_values in zip(positions, values):
                if len(seed_values) == 0:
                    continue
                jitter = np.linspace(-0.045, 0.045, len(seed_values))
                ax.scatter(pos + jitter, seed_values, s=30, color=color, alpha=0.45, edgecolors="none", zorder=3)
            ax.plot([], [], color=color, linewidth=6, alpha=0.55, label=MODEL_LABELS[model_name])
        ax.set_ylabel(ylabel, fontsize=BOXPLOT_FONT_SIZE, labelpad=22)
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{h:g}" for h in levels])
        ax.tick_params(axis="both", labelsize=BOXPLOT_FONT_SIZE)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Heterogeneity strength", fontsize=BOXPLOT_FONT_SIZE, labelpad=20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(BOXPLOT_MODELS), frameon=True, fontsize=BOXPLOT_FONT_SIZE)
    fig.suptitle("Sparse-trap DRL heterogeneity sweep: 30 seeds", fontsize=BOXPLOT_FONT_SIZE, y=0.97)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.91, wspace=0.2, hspace=0.10)
    fig.savefig(BOXPLOT_PATH, dpi=240)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    seed_from_existing_10seed()
    done = completed_keys()
    total = len(HETEROGENEITY_LEVELS) * len(SEEDS_30) * len(MODELS)
    count = len(done)
    for h in HETEROGENEITY_LEVELS:
        cfg = AdvantageTrapConfig(
            variant=variant_name(h),
            seeds=SEEDS_30,
            horizon=20,
            train_epochs=90,
            terminal_scale=16.0,
            trap_scale=5.0,
            actor_noise_std=0.20,
            heterogeneity=h,
            normalize_advantage=False,
        )
        for seed in SEEDS_30:
            for model_name in MODELS:
                key = (h, model_name, seed)
                if key in done:
                    continue
                result = train_one(model_name, seed, cfg)
                summary = result["summary"]
                summary["heterogeneity"] = h
                summary["model_label"] = MODEL_LABELS[model_name]
                summary["weight_mode"] = "random_conditioned"
                for row in result["curves"]:
                    row["heterogeneity"] = h
                    row["model_label"] = MODEL_LABELS[model_name]
                    row["weight_mode"] = "random_conditioned"
                append_csv(SUMMARY_PATH, [summary])
                append_csv(CURVES_PATH, result["curves"])
                done.add(key)
                count += 1
                print(f"[{count}/{total}] h={h:g} seed={seed} model={MODEL_LABELS[model_name]}", flush=True)
    df = write_summaries()
    plot_metric_boxplots(df)
    print(f"Wrote 30-seed sparse-trap heterogeneity sweep to {OUT_DIR}")


if __name__ == "__main__":
    main()
