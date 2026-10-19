from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs


MODELS = ("FC", "MH", "MoE", "MH+MoE")
GATE_MODELS = ("MoE", "MH+MoE")
LABELS = {name: name for name in MODELS}
COLORS = {
    "FC": "#3568a9",
    "MH": "#d17827",
    "MoE": "#2f8f53",
    "MH+MoE": "#b33b3b",
}
METRICS = (
    ("test_mse", "Critic MSE"),
    ("advantage_error_mae", "Advantage error\nMAE"),
    ("gradient_norm_mean", "Gradient\nnorm mean"),
)
OUTPUT_FILE = "controlled_critic_proposal_metric_boxplots.png"


def label_model(name: str) -> str:
    return LABELS.get(name, name)


def read_summary() -> pd.DataFrame:
    preferred = RESULTS_DIR / "strong_critic_regime_simulation_30seed_advantage_error_summary.csv"
    fallback = RESULTS_DIR / "strong_critic_regime_simulation_30seed_extended_grad_summary.csv"
    return pd.read_csv(preferred if preferred.exists() else fallback)


def plot_metric_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    levels: list[float],
    centers: np.ndarray,
    rng: np.random.Generator,
) -> None:
    offsets = dict(zip(MODELS, np.linspace(-0.35, 0.35, len(MODELS))))
    for model in MODELS:
        values = [
            df[(df["model"] == model) & (df["heterogeneity_strength"] == h)][metric].dropna().to_numpy()
            for h in levels
        ]
        positions = [center + offsets[model] for center in centers]
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.5},
            boxprops={"facecolor": COLORS[model], "edgecolor": COLORS[model], "alpha": 0.34, "linewidth": 1.4},
            whiskerprops={"color": COLORS[model], "linewidth": 1.2},
            capprops={"color": COLORS[model], "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.34)
        for pos, vals in zip(positions, values):
            if len(vals):
                jitter = rng.normal(0.0, 0.022, size=len(vals))
                ax.scatter(
                    pos + jitter,
                    vals,
                    s=34,
                    color=COLORS[model],
                    alpha=0.70,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=3,
                )
        ax.plot(
            [],
            [],
            color=COLORS[model],
            marker="s",
            linestyle="",
            markersize=15,
            alpha=0.58,
            label=label_model(model),
        )
    ax.set_ylabel(ylabel, fontsize=42, labelpad=12)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{h:g}" for h in levels], fontsize=40)
    ax.tick_params(axis="y", labelsize=38)
    ax.grid(axis="y", alpha=0.25)


def plot_gate_alignment(
    ax: plt.Axes,
    df: pd.DataFrame,
    levels: list[float],
    centers: np.ndarray,
    rng: np.random.Generator,
) -> None:
    width = 0.34
    offsets = {"MoE": -width / 2, "MH+MoE": width / 2}
    gate_df = df[df["model"].isin(GATE_MODELS)].copy()

    for model in GATE_MODELS:
        means = []
        stds = []
        values_by_h = []
        for h in levels:
            vals = gate_df[(gate_df["model"] == model) & (gate_df["heterogeneity_strength"] == h)][
                "gate_regime_alignment"
            ].dropna().to_numpy()
            values_by_h.append(vals)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))

        positions = centers + offsets[model]
        ax.bar(
            positions,
            means,
            width=width,
            color=COLORS[model],
            edgecolor=COLORS[model],
            alpha=0.34,
            linewidth=1.8,
            label=label_model(model),
            zorder=2,
        )
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="none",
            ecolor=COLORS[model],
            elinewidth=1.8,
            capsize=6,
            capthick=1.8,
            zorder=3,
        )
        for pos, vals in zip(positions, values_by_h):
            if len(vals):
                jitter = rng.normal(0.0, 0.025, size=len(vals))
                ax.scatter(
                    pos + jitter,
                    vals,
                    s=34,
                    color=COLORS[model],
                    alpha=0.70,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=4,
                )

    ax.set_ylabel("Gate-regime\nalignment", fontsize=42, labelpad=12)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{h:g}" for h in levels], fontsize=40)
    ax.tick_params(axis="y", labelsize=38)
    ax.set_ylim(0.0, 1.03)
    ax.grid(axis="y", alpha=0.25, zorder=1)


def plot_proposal_figure(df: pd.DataFrame) -> None:
    levels = sorted(df["heterogeneity_strength"].unique())
    centers = np.arange(len(levels), dtype=float) * 1.1
    rng = np.random.default_rng(90210)

    fig, axes = plt.subplots(1, 3, figsize=(37.5, 8.4), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.86, wspace=0.46)

    for ax, (metric, ylabel) in zip(axes, METRICS):
        plot_metric_boxplot(ax, df, metric, ylabel, levels, centers, rng)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=42,
    )
    fig.supxlabel("Heterogeneity strength h", fontsize=42, y=0.05)
    fig.savefig(FIGURES_DIR / OUTPUT_FILE, dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    df = read_summary()
    plot_proposal_figure(df)
    print(f"Wrote {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
