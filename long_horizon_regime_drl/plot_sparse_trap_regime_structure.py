"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from long_horizon_regime_drl.run_advantage_sparse_trap import GOOD_ACTIONS, REGIME_GOALS, TRAP_ACTIONS, AdvantageTrapConfig, regime_logits, softmax_np, step_env
FIG_DIR = Path(__file__).resolve().parent / 'figures'
REGIME_COLORS = np.array(['#2f7fb8', '#d77b2d', '#7a4fb3'])
REGIME_NAMES = ('Regime 1', 'Regime 2', 'Regime 3')

def state_grid(load: float, n: int=160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-0.85, 0.85, n)
    y = np.linspace(-0.85, 0.85, n)
    xx, yy = np.meshgrid(x, y)
    state = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size), np.full(xx.size, 0.45), np.full(xx.size, load)], axis=1).astype(np.float32)
    return (xx, yy, state)

def regime_rgb(prob: np.ndarray) -> np.ndarray:
    rgb = np.array([mcolors.to_rgb(c) for c in REGIME_COLORS])
    return prob @ rgb

def plot_regime_regions() -> None:
    loads = (0.2, 0.5, 0.8)
    fig = plt.figure(figsize=(13.0, 11.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.68, 0.92])
    for col, load in enumerate(loads):
        ax = fig.add_subplot(gs[0, col])
        xx, yy, state = state_grid(load)
        probs = softmax_np(regime_logits(state))
        dominant = probs.argmax(axis=1).reshape(xx.shape)
        confidence = probs.max(axis=1).reshape(xx.shape)
        rgb = regime_rgb(probs).reshape(xx.shape + (3,))
        ax.imshow(rgb, extent=[xx.min(), xx.max(), yy.min(), yy.max()], origin='lower', interpolation='bilinear', aspect='equal')
        ax.contour(xx, yy, dominant, levels=[0.5, 1.5], colors='white', linewidths=1.0, alpha=0.9)
        ax.contour(xx, yy, confidence, levels=[0.55, 0.7, 0.85], colors='black', linewidths=0.55, alpha=0.45)
        for i, goal in enumerate(REGIME_GOALS):
            ax.scatter(goal[0], goal[1], s=95, marker='*', color=REGIME_COLORS[i], edgecolor='black', linewidth=0.8)
        ax.set_title(f'Latent regime probabilities, load={load:.2f}')
        ax.set_xlabel('state x')
        if col == 0:
            ax.set_ylabel('state y')
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
    ax = fig.add_subplot(gs[1, :2])
    matrix = np.zeros((3, 8), dtype=float)
    for regime in range(3):
        matrix[regime, GOOD_ACTIONS[regime]] = 1.0
        matrix[regime, TRAP_ACTIONS[regime]] = -1.0
    cmap = mcolors.ListedColormap(['#b33b3b', '#e8e8e8', '#2f8f53'])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f'a{i}' for i in range(8)])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(REGIME_NAMES)
    ax.set_title('Same action has different meaning by regime')
    for regime in range(3):
        for action in range(8):
            label = 'good' if matrix[regime, action] > 0 else 'trap' if matrix[regime, action] < 0 else 'neutral'
            ax.text(action, regime, label, ha='center', va='center', fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    handles = [plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=REGIME_COLORS[i], markersize=12, label=REGIME_NAMES[i]) for i in range(3)]
    handles.extend([plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='#2f8f53', markersize=12, label='good action'), plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='#b33b3b', markersize=12, label='trap action'), plt.Line2D([0], [0], marker='*', color='black', markerfacecolor='white', markersize=12, label='regime goal')])
    ax.legend(handles=handles, loc='center', frameon=False)
    ax.axis('off')
    cfg = AdvantageTrapConfig(variant='visualize_sparse_trap', horizon=20, train_epochs=90, heterogeneity=5.0)
    x = np.linspace(-0.75, 0.75, 14)
    y = np.linspace(-0.75, 0.75, 14)
    xx, yy = np.meshgrid(x, y)
    base_state = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size), np.full(xx.size, 0.45), np.full(xx.size, 0.55)], axis=1).astype(np.float32)
    for regime in range(3):
        ax = fig.add_subplot(gs[2, regime])
        action = int(GOOD_ACTIONS[regime][0])
        regimes = np.full(base_state.shape[0], regime, dtype=np.int64)
        actions = np.full(base_state.shape[0], action, dtype=np.int64)
        next_state, _ = step_env(base_state.copy(), regimes, actions, 0, cfg, None, add_noise=False)
        delta = next_state[:, :2] - base_state[:, :2]
        ax.quiver(base_state[:, 0], base_state[:, 1], delta[:, 0], delta[:, 1], angles='xy', scale_units='xy', scale=1.0, width=0.004, color=REGIME_COLORS[regime], alpha=0.85)
        ax.scatter(REGIME_GOALS[regime, 0], REGIME_GOALS[regime, 1], s=115, marker='*', color=REGIME_COLORS[regime], edgecolor='black', linewidth=0.8)
        ax.set_title(f'{REGIME_NAMES[regime]} dynamics for good action a{action}')
        ax.set_xlabel('state x')
        if regime == 0:
            ax.set_ylabel('state y')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.2, 1.0)
        ax.grid(alpha=0.22)
    fig.suptitle('Sparse-trap control problem exposes three latent regimes', fontsize=15)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'sparse_trap_multiregime_structure.png', dpi=240)
    plt.close(fig)

if __name__ == '__main__':
    plot_regime_regions()
