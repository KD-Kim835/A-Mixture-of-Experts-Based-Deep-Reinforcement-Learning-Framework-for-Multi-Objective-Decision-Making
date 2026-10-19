"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import FIGURES_DIR, TrainConfig
from models import build_model, count_parameters
from strong_critic_simulation import StrongCriticConfig, generate_dataset, regime_logits, softmax_np, split_dataset
from train import fit_model, resolve_device, set_seed
LABELS = {'MoE': 'MoE', 'MH+MoE': 'MH+MoE'}
HETEROGENEITY = 5.0
SEED = 11
GRID_SIZE = 180

def train_gate_model(model_name: str, cfg: StrongCriticConfig, train_cfg: TrainConfig) -> torch.nn.Module:
    set_seed(SEED)
    dataset = generate_dataset(cfg, SEED, HETEROGENEITY)
    split = split_dataset(dataset, cfg, SEED)
    model = build_model(model_name, split['train']['X'].shape[1], cfg.n_objectives, train_cfg.hidden_dim, train_cfg.n_experts)
    result = fit_model(model, split['train'], split['test'], train_cfg, SEED)
    trained = result['model']
    print(f"{LABELS[model_name]}: test_mse={result['test_mse']:.4f}, parameters={count_parameters(trained)}", flush=True)
    return trained

def make_grid(cfg: StrongCriticConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    obs = np.column_stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float32), np.full(xx.size, 0.5, dtype=np.float32), np.full(xx.size, 0.5, dtype=np.float32)]).astype(np.float32)
    action_zero = np.zeros((obs.shape[0], cfg.n_actions), dtype=np.float32)
    action_zero[:, 0] = 1.0
    x_full = np.concatenate([obs, action_zero], axis=1).astype(np.float32)
    return (xx, yy, obs, x_full)

@torch.no_grad()
def gate_grid(model: torch.nn.Module, x_full: np.ndarray, device: torch.device) -> np.ndarray:
    gates = model.gate_weights(torch.from_numpy(x_full).to(device)).detach().cpu().numpy()
    return gates.reshape(GRID_SIZE, GRID_SIZE, -1)

def plot_probability_maps(true_probs: np.ndarray, model_probs: dict[str, np.ndarray]) -> None:
    row_titles = ['True latent regime probability', 'MoE softmax gate', 'MH+MoE softmax gate']
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 15.5), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.88, wspace=0.28, hspace=0.34)
    maps = [true_probs, model_probs['MoE'], model_probs['MH+MoE']]
    for row, probs in enumerate(maps):
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(probs[:, :, col], origin='lower', extent=[-1, 1, -1, 1], cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            if row == 0:
                ax.set_title(f'Regime {col + 1}', fontsize=22, pad=14)
            else:
                ax.set_title(f'Expert {col + 1}', fontsize=22, pad=14)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel('y', fontsize=16)
            ax.tick_params(labelsize=13)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=12)
    for row, label in enumerate(row_titles):
        bbox = axes[row, 0].get_position()
        fig.text(0.035, (bbox.y0 + bbox.y1) / 2, label, rotation=90, va='center', ha='center', fontsize=22)
    fig.suptitle(f'Softmax gate responsibility maps at h={HETEROGENEITY:g}, seed={SEED}', fontsize=24)
    fig.savefig(FIGURES_DIR / f'controlled_critic_gate_softmax_responsibility_colormap_h{HETEROGENEITY:g}.png', dpi=220, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg = StrongCriticConfig()
    train_cfg = TrainConfig(epochs=85, batch_size=192, gate_entropy_weight=0.001, load_balance_weight=0.003)
    _, _, obs, x_full = make_grid(cfg)
    true_probs = softmax_np(regime_logits(obs, cfg.regime_separability)).reshape(GRID_SIZE, GRID_SIZE, 3)
    device = resolve_device(train_cfg.device)
    learned = {name: gate_grid(train_gate_model(name, cfg, train_cfg), x_full, device) for name in LABELS}
    plot_probability_maps(true_probs, learned)

if __name__ == '__main__':
    main()
