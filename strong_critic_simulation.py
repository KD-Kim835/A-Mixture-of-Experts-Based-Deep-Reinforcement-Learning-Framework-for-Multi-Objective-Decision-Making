"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from config import TrainConfig
from diagnostics import gate_diagnostics, gradient_interference
from models import build_model, count_parameters
from train import fit_model, resolve_device, set_seed
WEIGHTS = np.array([0.45, 0.35, 0.2], dtype=np.float32)

@dataclass(frozen=True)
class StrongCriticConfig:
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
    heterogeneity_levels: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    n_samples: int = 1800
    test_fraction: float = 0.3
    n_actions: int = 8
    n_objectives: int = 3
    regime_separability: float = 4.2
    noise_std: tuple[float, float, float] = (0.035, 0.04, 0.04)

def sample_states(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    y = rng.uniform(-1.0, 1.0, size=(n, 1))
    vx = rng.normal(0.0, 0.65, size=(n, 1))
    risk = rng.beta(2.0, 2.0, size=(n, 1))
    load = rng.beta(2.0, 2.0, size=(n, 1))
    return np.concatenate([x, y, vx, risk, load], axis=1).astype(np.float32)

def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def regime_logits(obs: np.ndarray, separability: float) -> np.ndarray:
    x, y, vx, risk, load = [obs[:, i] for i in range(5)]
    radial = np.sqrt((x + 0.05) ** 2 + (y - 0.05) ** 2)
    logits = np.stack([2.2 * np.tanh(8.0 * (x + 0.28 * y - 0.18)) - 1.2 * np.tanh(7.0 * (risk - 0.58)), -2.3 * np.tanh(8.0 * (x - 0.2 * y + 0.12)) + 1.1 * np.tanh(7.0 * (risk - 0.48)), 2.0 * np.tanh(7.0 * (load - 0.48)) + 1.7 * np.tanh(7.0 * (0.55 - radial)) + 0.8 * np.tanh(2.0 * np.abs(vx))], axis=1)
    return separability * logits

def sample_regimes(obs: np.ndarray, cfg: StrongCriticConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    probs = softmax_np(regime_logits(obs, cfg.regime_separability)).astype(np.float32)
    regimes = np.array([rng.choice(3, p=p) for p in probs], dtype=np.int64)
    return (regimes, probs)

def make_inputs(obs: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    one_hot = np.eye(n_actions, dtype=np.float32)[actions]
    return np.concatenate([obs, one_hot], axis=1).astype(np.float32)

def shared_q(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    x, y, vx, risk, load = [obs[:, i] for i in range(5)]
    energy = np.array([0.7, 0.8, 0.95, 1.05, 1.15, 1.0, 1.25, 1.35], dtype=np.float32)[actions]
    return np.stack([0.06 * x + 0.04 * y - 0.03 * energy, -0.05 * risk - 0.03 * energy, -0.05 * load - 0.03 * np.abs(vx) - 0.03 * energy], axis=1).astype(np.float32)

def strong_regime_q(obs: np.ndarray, actions: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    x, y, vx, risk, load = [obs[:, i] for i in range(5)]
    sign_x = np.where(x >= 0.0, 1.0, -1.0)
    sign_y = np.where(y >= 0.0, 1.0, -1.0)
    near_boundary = (np.abs(x + 0.25 * y) < 0.18).astype(np.float32)
    high_load = (load > 0.52).astype(np.float32)
    a0_q1 = np.array([2.8, 2.2, -2.2, -2.6, -1.4, -1.7, -3.0, -3.2], dtype=np.float32)[actions]
    a0_q2 = np.array([2.0, 2.5, -1.8, -2.2, -1.2, -1.5, -2.6, -2.8], dtype=np.float32)[actions]
    a0_q3 = np.array([1.8, 2.0, -1.5, -1.9, -1.0, -1.3, -2.4, -2.6], dtype=np.float32)[actions]
    a1_q1 = np.array([-2.8, -2.4, 2.6, 2.2, 1.5, -1.3, -2.3, -2.7], dtype=np.float32)[actions]
    a1_q2 = np.array([-2.2, -1.9, -2.8, 2.9, 2.3, -1.1, -2.0, -2.4], dtype=np.float32)[actions]
    a1_q3 = np.array([-2.0, -1.7, 2.2, -2.7, 2.4, -0.9, -1.8, -2.1], dtype=np.float32)[actions]
    a2_low_q1 = np.array([-2.0, -1.7, -1.6, -1.3, -0.8, 2.4, 2.9, 2.2], dtype=np.float32)[actions]
    a2_high_q1 = np.array([2.5, 2.1, -1.8, -2.1, -1.4, -2.5, -2.9, -3.2], dtype=np.float32)[actions]
    a2_low_q2 = np.array([-1.5, -1.4, -2.0, -1.6, -1.1, 1.9, 2.4, 2.0], dtype=np.float32)[actions]
    a2_high_q2 = np.array([2.6, 2.3, -2.1, -2.5, -1.7, -2.3, -2.8, -3.0], dtype=np.float32)[actions]
    a2_q3 = np.array([-2.1, -1.8, -1.2, -0.8, 1.0, 2.3, 2.8, 2.5], dtype=np.float32)[actions]
    r0_q1 = a0_q1 + 2.3 * x + 1.6 * y - 1.4 * risk - 2.0 * near_boundary
    r0_q2 = a0_q2 + 1.9 * x - 1.8 * risk + 0.9 * sign_y - 1.2 * high_load
    r0_q3 = a0_q3 + 1.6 * y - 1.4 * load - 0.7 * np.abs(vx) + 0.7 * sign_x
    r1_q1 = a1_q1 - 2.4 * x - 1.4 * y + 1.1 * vx + 1.6 * near_boundary
    r1_q2 = a1_q2 - 2.1 * x + 1.8 * y + 1.6 * risk - 1.1 * load
    r1_q3 = a1_q3 - 1.8 * x + 1.3 * y + 1.0 * np.abs(vx) - 1.2 * load - 0.8 * sign_x
    r2_q1 = (1.0 - high_load) * a2_low_q1 + high_load * a2_high_q1 - 1.5 * x + 1.4 * sign_y - 2.0 * near_boundary
    r2_q2 = (1.0 - high_load) * a2_low_q2 + high_load * a2_high_q2 - 2.3 * load - 1.2 * risk + 1.4 * sign_x
    r2_q3 = a2_q3 + 2.1 * (np.sqrt(x ** 2 + y ** 2) < 0.55).astype(np.float32) - 1.7 * np.abs(vx) + 1.2 * (y < -0.25)
    all_q = np.stack([np.stack([r0_q1, r0_q2, r0_q3], axis=1), np.stack([r1_q1, r1_q2, r1_q3], axis=1), np.stack([r2_q1, r2_q2, r2_q3], axis=1)], axis=1).astype(np.float32)
    return all_q[np.arange(obs.shape[0]), regimes]

def compute_q(obs: np.ndarray, actions: np.ndarray, regimes: np.ndarray, heterogeneity: float, rng: np.random.Generator | None, cfg: StrongCriticConfig) -> np.ndarray:
    q = 0.12 * shared_q(obs, actions) + heterogeneity * strong_regime_q(obs, actions, regimes)
    if rng is not None:
        noise = np.stack([rng.normal(0.0, s, size=q.shape[0]) for s in cfg.noise_std], axis=1).astype(np.float32)
        q = q + noise
    return q.astype(np.float32)

def generate_dataset(cfg: StrongCriticConfig, seed: int, heterogeneity: float) -> dict:
    rng = np.random.default_rng(seed)
    obs = sample_states(rng, cfg.n_samples)
    regimes, probs = sample_regimes(obs, cfg, rng)
    actions = rng.integers(0, cfg.n_actions, size=cfg.n_samples, dtype=np.int64)
    q = compute_q(obs, actions, regimes, heterogeneity, rng, cfg)
    x = make_inputs(obs, actions, cfg.n_actions)
    return {'X': torch.from_numpy(x), 'Q': torch.from_numpy(q), 'obs': obs, 'actions': actions, 'ctx_true': regimes, 'regime_probs': probs}

def split_dataset(dataset: dict, cfg: StrongCriticConfig, seed: int) -> dict[str, dict]:
    rng = np.random.default_rng(seed + 101)
    idx = rng.permutation(len(dataset['actions']))
    n_test = int(round(cfg.test_fraction * len(idx)))
    out = {}
    for name, selected in (('test', idx[:n_test]), ('train', idx[n_test:])):
        out[name] = {key: value[selected] for key, value in dataset.items()}
    return out

def enumerate_inputs(obs: np.ndarray, cfg: StrongCriticConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repeated_obs = np.repeat(obs, cfg.n_actions, axis=0)
    actions = np.tile(np.arange(cfg.n_actions, dtype=np.int64), obs.shape[0])
    return (make_inputs(repeated_obs, actions, cfg.n_actions), repeated_obs, actions)

@torch.no_grad()
def action_metrics(model: torch.nn.Module, obs: np.ndarray, regimes: np.ndarray, heterogeneity: float, cfg: StrongCriticConfig, device: torch.device) -> dict[str, float]:
    x_all, repeated_obs, actions = enumerate_inputs(obs, cfg)
    repeated_regimes = np.repeat(regimes, cfg.n_actions, axis=0)
    true_q = compute_q(repeated_obs, actions, repeated_regimes, heterogeneity, None, cfg)
    pred_q = model(torch.from_numpy(x_all).to(device)).detach().cpu().numpy()
    true_scalar = true_q @ WEIGHTS
    pred_scalar = pred_q @ WEIGHTS
    true_by_state = true_scalar.reshape(obs.shape[0], cfg.n_actions)
    pred_by_state = pred_scalar.reshape(obs.shape[0], cfg.n_actions)
    chosen = pred_by_state.argmax(axis=1)
    optimal = true_by_state.argmax(axis=1)
    pred_adv = pred_by_state - pred_by_state.mean(axis=1, keepdims=True)
    true_adv = true_by_state - true_by_state.mean(axis=1, keepdims=True)
    advantage_error = (pred_adv - true_adv).reshape(-1)
    return {'scalarized_value_mse': float(np.mean((true_scalar - pred_scalar) ** 2)), 'action_regret': float(np.mean(true_by_state.max(axis=1) - true_by_state[np.arange(obs.shape[0]), chosen])), 'optimal_action_match': float((chosen == optimal).mean()), 'advantage_error_variance': float(np.var(advantage_error)), 'advantage_error_mae': float(np.mean(np.abs(advantage_error))), 'advantage_error_rmse': float(np.sqrt(np.mean(advantage_error ** 2)))}

def run_one(model_name: str, seed: int, heterogeneity: float, cfg: StrongCriticConfig, train_cfg: TrainConfig) -> dict:
    set_seed(seed)
    dataset = generate_dataset(cfg, seed, heterogeneity)
    split = split_dataset(dataset, cfg, seed)
    model = build_model(model_name, split['train']['X'].shape[1], cfg.n_objectives, train_cfg.hidden_dim, train_cfg.n_experts)
    result = fit_model(model, split['train'], split['test'], train_cfg, seed)
    trained = result['model']
    device = resolve_device(train_cfg.device)
    grad = gradient_interference(trained, split['test']['X'].to(device), split['test']['Q'].to(device), max_samples=min(512, split['test']['X'].shape[0]))
    gate_summary, _ = gate_diagnostics(trained, split['test']['X'].to(device), split['test']['ctx_true'])
    actor = action_metrics(trained, split['test']['obs'][:600], split['test']['ctx_true'][:600], heterogeneity, cfg, device)
    return {'model': model_name, 'seed': seed, 'heterogeneity_strength': heterogeneity, 'parameter_count': count_parameters(trained), 'test_mse': result['test_mse'], **grad, **gate_summary, **actor}
