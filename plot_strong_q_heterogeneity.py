"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import FIGURES_DIR
from strong_critic_simulation import WEIGHTS, StrongCriticConfig, compute_q
REFERENCE_LEVELS = (2.0, 5.0)
SPREAD_LEVELS = (0.0, 2.0, 3.0, 4.0, 5.0)
OBJECTIVE_NAMES = ('Q1', 'Q2', 'Q3')

def grid_states(n: int=90, vx: float=0.0, risk: float=0.5, load: float=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    obs = np.stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, vx), np.full(xx.size, risk), np.full(xx.size, load)], axis=1).astype(np.float32)
    return (xx, yy, obs)

def q_for_regime_actions(obs: np.ndarray, regime: int, h: float, cfg: StrongCriticConfig) -> np.ndarray:
    repeated_obs = np.repeat(obs, cfg.n_actions, axis=0)
    actions = np.tile(np.arange(cfg.n_actions, dtype=np.int64), obs.shape[0])
    regimes = np.full(repeated_obs.shape[0], regime, dtype=np.int64)
    q = compute_q(repeated_obs, actions, regimes, h, None, cfg)
    return q.reshape(obs.shape[0], cfg.n_actions, cfg.n_objectives)

def q_for_fixed_action(obs: np.ndarray, regime: int, action: int, h: float, cfg: StrongCriticConfig) -> np.ndarray:
    actions = np.full(obs.shape[0], action, dtype=np.int64)
    regimes = np.full(obs.shape[0], regime, dtype=np.int64)
    return compute_q(obs, actions, regimes, h, None, cfg)

def h_slug(h: float) -> str:
    return f'{h:g}'.replace('.', 'p')

def plot_regime_value_spread(cfg: StrongCriticConfig) -> None:
    xx, yy, obs = grid_states()
    style = {'font.size': 25, 'axes.titlesize': 25, 'axes.labelsize': 25, 'xtick.labelsize': 25, 'ytick.labelsize': 25, 'figure.titlesize': 30}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(SPREAD_LEVELS), figsize=(32.0, 7.2), sharex=True, sharey=True, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.22, wspace=0.14, hspace=0.08)
        flat_axes = axes.ravel()
        vmax = None
        maps = []
        for h in SPREAD_LEVELS:
            scalar_by_regime = []
            for regime in range(3):
                q = q_for_regime_actions(obs, regime, h, cfg)
                scalar_by_regime.append((q @ WEIGHTS).max(axis=1))
            scalar_by_regime = np.stack(scalar_by_regime, axis=1)
            spread = scalar_by_regime.std(axis=1).reshape(xx.shape)
            maps.append(spread)
            vmax = max(float(spread.max()), vmax or 0.0)
        for ax, h, spread in zip(flat_axes, SPREAD_LEVELS, maps):
            im = ax.imshow(spread, origin='lower', extent=[-1, 1, -1, 1], cmap='magma', vmin=0.0, vmax=vmax, aspect='equal')
            ax.set_title(f'h={h:g}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4.4%', pad=0.07)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=25, pad=1)
        fig.suptitle('Generated Q heterogeneity: regime value spread over x-y')
        fig.savefig(FIGURES_DIR / 'strong_q_heterogeneity_regime_value_spread.png', dpi=220, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)

def plot_reference_style_q_by_h(cfg: StrongCriticConfig) -> None:
    action = 2
    xx, yy, obs = grid_states(n=80, vx=0.0, risk=0.5, load=0.5)
    objective_limits = []
    for obj in range(cfg.n_objectives):
        maps = []
        for h in REFERENCE_LEVELS:
            for regime in range(3):
                q = q_for_fixed_action(obs, regime, action, h, cfg)
                maps.append(q[:, obj])
        values = np.concatenate(maps)
        objective_limits.append((float(values.min()), float(values.max())))
    style = {'font.size': 25, 'axes.titlesize': 25, 'axes.labelsize': 25, 'legend.fontsize': 25, 'xtick.labelsize': 25, 'ytick.labelsize': 25, 'figure.titlesize': 30}
    with plt.rc_context(style):
        for h in REFERENCE_LEVELS:
            fig, heat_axes = plt.subplots(3, 3, figsize=(17.0, 15.2), sharex=True, sharey=True, constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.22, wspace=0.12, hspace=0.18)
            heat_maps = []
            for regime in range(3):
                q = q_for_fixed_action(obs, regime, action, h, cfg)
                heat_maps.append([q[:, obj].reshape(xx.shape) for obj in range(cfg.n_objectives)])
            for obj in range(cfg.n_objectives):
                vmin, vmax = objective_limits[obj]
                for regime in range(3):
                    ax = heat_axes[regime][obj]
                    im = ax.imshow(heat_maps[regime][obj], origin='lower', extent=[-1, 1, -1, 1], cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
                    if regime == 0:
                        ax.set_title(f'{OBJECTIVE_NAMES[obj]}')
                    if obj == 0:
                        ax.set_ylabel(f'Regime {regime}\ny')
                    else:
                        ax.set_ylabel('')
                    if regime == 2:
                        ax.set_xlabel('x')
                    else:
                        ax.set_xlabel('')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='4.2%', pad=0.1)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.ax.tick_params(labelsize=25, pad=1)
            fig.suptitle(f'Generated True Q Heterogeneity at h={h:g}, action A{action}')
            fig.savefig(FIGURES_DIR / f'strong_q_reference_style_h_{h_slug(h)}.png', dpi=220, bbox_inches='tight', pad_inches=0.25)
            plt.close(fig)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg = StrongCriticConfig()
    plot_reference_style_q_by_h(cfg)
    plot_regime_value_spread(cfg)

if __name__ == '__main__':
    main()
