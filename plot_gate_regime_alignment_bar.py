"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import FIGURES_DIR
MODELS = ('MoE', 'MH+MoE')
LABELS = {name: name for name in MODELS}
COLORS = {'MoE': '#2f8f53', 'MH+MoE': '#b33b3b'}

def plot_gate_alignment(df: pd.DataFrame) -> None:
    df = df[df['model'].isin(MODELS)].copy()
    levels = sorted(df['heterogeneity_strength'].unique())
    centers = np.arange(len(levels), dtype=float)
    width = 0.34
    offsets = {'MoE': -width / 2, 'MH+MoE': width / 2}
    rng = np.random.default_rng(517)
    fig, ax = plt.subplots(figsize=(15.0, 9.0))
    for model in MODELS:
        means = []
        stds = []
        values_by_h = []
        for h in levels:
            vals = df[(df['model'] == model) & (df['heterogeneity_strength'] == h)]['gate_regime_alignment'].dropna().to_numpy()
            values_by_h.append(vals)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        positions = centers + offsets[model]
        ax.bar(positions, means, width=width, color=COLORS[model], edgecolor=COLORS[model], alpha=0.32, linewidth=2.0, label=LABELS[model], zorder=2)
        ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor=COLORS[model], elinewidth=2.0, capsize=7, capthick=2.0, zorder=3)
        for pos, vals in zip(positions, values_by_h):
            jitter = rng.normal(0.0, 0.025, size=len(vals))
            ax.scatter(pos + jitter, vals, s=42, color=COLORS[model], alpha=0.62, edgecolors='white', linewidths=0.35, zorder=4)
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{h:g}' for h in levels], fontsize=28)
    ax.set_xlabel('Heterogeneity strength h', fontsize=30, labelpad=16)
    ax.set_ylabel('Gate-regime alignment', fontsize=30, labelpad=16)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis='y', alpha=0.25, zorder=1)
    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=28)
    ax.set_title('Gate-regime alignment across heterogeneity levels', fontsize=32, pad=22)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'controlled_critic_gate_regime_alignment_bar.png', dpi=220, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

def main() -> None:
    from config import RESULTS_DIR, ensure_output_dirs
    ensure_output_dirs()
    plot_gate_alignment(pd.read_csv(RESULTS_DIR / 'strong_critic_regime_simulation_30seed_advantage_error_summary.csv'))

if __name__ == '__main__':
    main()
