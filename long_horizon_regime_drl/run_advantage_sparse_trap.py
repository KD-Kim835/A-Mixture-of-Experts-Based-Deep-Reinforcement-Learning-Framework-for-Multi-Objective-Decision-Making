"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from actor_critic_experiment import build_actor_critic_critic, build_actor_policy, critic_forward
from config import TrainConfig
from models import count_parameters
from train import regularization_loss, resolve_device, set_seed
N_ACTIONS = 8
N_OBJECTIVES = 3
OBS_DIM = 6
EVAL_WEIGHTS = {'fixed_45_35_20': np.array([0.45, 0.35, 0.2], dtype=np.float32), 'equal': np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), 'q1_heavy': np.array([0.7, 0.2, 0.1], dtype=np.float32), 'q2_heavy': np.array([0.2, 0.7, 0.1], dtype=np.float32), 'q3_heavy': np.array([0.2, 0.1, 0.7], dtype=np.float32)}
PREFIXES = tuple(EVAL_WEIGHTS.keys())
GOOD_ACTIONS = {0: np.array([0, 1], dtype=np.int64), 1: np.array([2, 3], dtype=np.int64), 2: np.array([5, 6], dtype=np.int64)}
TRAP_ACTIONS = {0: np.array([6, 7], dtype=np.int64), 1: np.array([0, 1], dtype=np.int64), 2: np.array([2, 3], dtype=np.int64)}
ACTION_EFFECTS = np.array([[0.18, 0.1, 0.1, -0.06, -0.05], [0.12, 0.18, 0.1, -0.05, -0.06], [-0.18, 0.1, 0.1, -0.06, -0.05], [-0.12, 0.18, 0.1, -0.05, -0.06], [0.0, -0.1, -0.05, 0.02, -0.01], [-0.1, -0.18, 0.1, -0.05, -0.06], [0.1, -0.18, 0.1, -0.06, -0.05], [0.2, -0.06, -0.06, 0.05, 0.05]], dtype=np.float32)
REGIME_GOALS = np.array([[1.05, 0.85], [-1.05, 0.85], [0.0, -1.1]], dtype=np.float32)
MODELS = ('FC', 'MH', 'ActorMoE', 'MoE', 'MH+MoE')
COLORS = {'FC': '#3568a9', 'MH': '#d17827', 'ActorMoE': '#9467bd', 'MoE': '#2f8f53', 'MH+MoE': '#b33b3b'}

@dataclass(frozen=True)
class AdvantageTrapConfig:
    variant: str
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    horizon: int = 15
    train_epochs: int = 80
    batch_episodes: int = 192
    eval_episodes: int = 800
    mc_eval_episodes: int = 192
    gamma: float = 0.97
    heterogeneity: float = 3.0
    actor_lr: float = 0.0007
    critic_lr: float = 0.001
    entropy_weight: float = 0.012
    hidden_dim: int = 96
    actor_noise_std: float = 0.16
    critic_noise_std: float = 0.015
    transition_noise_std: float = 0.012
    reward_noise_std: float = 0.03
    terminal_scale: float = 12.0
    trap_scale: float = 4.0
    step_trap_penalty: float = 0.03
    shaping_scale: float = 0.0
    normalize_advantage: bool = False
    dirichlet_alpha: float = 1.0
    device: str = 'auto'

def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def regime_logits(state: np.ndarray) -> np.ndarray:
    x, y, progress, risk, load = [state[:, i] for i in range(5)]
    return np.stack([3.2 * np.tanh(3.0 * x + 0.7 * y) - 0.6 * risk + 0.2 * progress, -3.1 * np.tanh(3.0 * x - 0.6 * y) + 0.8 * risk - 0.4 * load, 2.8 * np.tanh(2.7 * (load - 0.45)) - 1.2 * y + 0.6 * np.sin(3.2 * x)], axis=1)

def initial_state(batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    state = np.stack([rng.uniform(-0.65, 0.65, batch_size), rng.uniform(-0.65, 0.65, batch_size), rng.normal(0.0, 0.08, batch_size), rng.beta(2.1, 2.5, batch_size), rng.beta(2.0, 2.4, batch_size)], axis=1).astype(np.float32)
    probs = softmax_np(regime_logits(state))
    regime = np.array([rng.choice(3, p=p) for p in probs], dtype=np.int64)
    return (state, regime)

def actor_observe(state: np.ndarray, t: int, cfg: AdvantageTrapConfig, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, cfg.actor_noise_std, size=(state.shape[0], 4)).astype(np.float32)
    partial = np.stack([state[:, 0], state[:, 1], state[:, 3], state[:, 4]], axis=1) + noise
    time_left = np.full((state.shape[0], 1), 1.0 - t / cfg.horizon, dtype=np.float32)
    ambiguity = (0.35 * np.sin(2.2 * state[:, 0]) + 0.35 * np.cos(2.0 * state[:, 1])).reshape(-1, 1).astype(np.float32)
    return np.concatenate([partial, time_left, ambiguity], axis=1).astype(np.float32)

def critic_observe(state: np.ndarray, t: int, cfg: AdvantageTrapConfig, rng: np.random.Generator | None) -> np.ndarray:
    obs = state.copy()
    if rng is not None and cfg.critic_noise_std > 0:
        obs += rng.normal(0.0, cfg.critic_noise_std, size=obs.shape).astype(np.float32)
    time_left = np.full((state.shape[0], 1), 1.0 - t / cfg.horizon, dtype=np.float32)
    return np.concatenate([obs, time_left], axis=1).astype(np.float32)

def sample_weights(batch_size: int, cfg: AdvantageTrapConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.full(3, cfg.dirichlet_alpha), size=batch_size).astype(np.float32)

def actor_input(obs: np.ndarray, weights: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([obs, weights], axis=1).astype(np.float32))

def action_score(actions: np.ndarray, regime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = np.zeros(actions.shape[0], dtype=bool)
    trap = np.zeros(actions.shape[0], dtype=bool)
    for r in range(3):
        idx = regime == r
        good[idx] = np.isin(actions[idx], GOOD_ACTIONS[r])
        trap[idx] = np.isin(actions[idx], TRAP_ACTIONS[r])
    neutral = ~(good | trap)
    score = np.zeros(actions.shape[0], dtype=np.float32)
    score[good] = 1.0
    score[neutral] = -0.15
    score[trap] = -1.15
    return (score, good, trap)

def step_env(state: np.ndarray, regime: np.ndarray, actions: np.ndarray, t: int, cfg: AdvantageTrapConfig, rng: np.random.Generator | None, add_noise: bool) -> tuple[np.ndarray, np.ndarray]:
    score, good, trap = action_score(actions, regime)
    effects = ACTION_EFFECTS[actions].copy()
    goals = REGIME_GOALS[regime]
    direction = goals - state[:, :2]
    effects[:, :2] += 0.055 * cfg.heterogeneity * score[:, None] * direction
    effects[:, 2] += 0.055 * cfg.heterogeneity * score
    effects[:, 3] += -0.035 * cfg.heterogeneity * score
    effects[:, 4] += -0.03 * cfg.heterogeneity * score
    effects[trap, 2] -= 0.12 * cfg.trap_scale
    effects[trap, 3] += 0.075 * cfg.trap_scale
    effects[trap, 4] += 0.07 * cfg.trap_scale
    next_state = state + effects
    if rng is not None:
        next_state += rng.normal(0.0, cfg.transition_noise_std, size=next_state.shape).astype(np.float32)
    next_state[:, :2] = np.clip(next_state[:, :2], -1.55, 1.55)
    next_state[:, 2] = np.clip(next_state[:, 2], -2.5, 2.5)
    next_state[:, 3:] = np.clip(next_state[:, 3:], 0.0, 2.0)
    dist = np.linalg.norm(next_state[:, :2] - goals, axis=1)
    success = 1.0 / (1.0 + np.exp(8.0 * (dist - 0.4)))
    safe = 1.0 / (1.0 + np.exp(9.0 * (next_state[:, 3] - 0.52)))
    stable = 1.0 / (1.0 + np.exp(9.0 * (next_state[:, 4] - 0.52)))
    progress = np.maximum(next_state[:, 2], -1.5)
    reward = cfg.shaping_scale * np.stack([0.6 * score - 0.25 * dist, 0.5 * score - 0.3 * next_state[:, 3], 0.5 * score - 0.3 * next_state[:, 4]], axis=1).astype(np.float32)
    reward[trap] -= cfg.step_trap_penalty * cfg.trap_scale
    if t == cfg.horizon - 1:
        terminal = np.stack([cfg.terminal_scale * (success + 0.35 * progress) - cfg.trap_scale * np.maximum(0.45 - success, 0.0), cfg.terminal_scale * (safe + 0.2 * success) - cfg.trap_scale * np.maximum(next_state[:, 3] - 0.6, 0.0), cfg.terminal_scale * (stable + 0.2 * success) - cfg.trap_scale * np.maximum(next_state[:, 4] - 0.6, 0.0)], axis=1)
        reward += terminal.astype(np.float32)
    if add_noise and rng is not None:
        reward += rng.normal(0.0, cfg.reward_noise_std, size=reward.shape).astype(np.float32)
    return (next_state.astype(np.float32), reward.astype(np.float32))

def one_hot(actions: np.ndarray) -> np.ndarray:
    return np.eye(N_ACTIONS, dtype=np.float32)[actions]

def critic_inputs(obs: np.ndarray, actions: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([obs, one_hot(actions)], axis=1).astype(np.float32))

def all_action_critic_inputs(obs: np.ndarray) -> torch.Tensor:
    repeated = np.repeat(obs, N_ACTIONS, axis=0)
    actions = np.tile(np.arange(N_ACTIONS, dtype=np.int64), obs.shape[0])
    return critic_inputs(repeated, actions)

def critic_all_q(critic: torch.nn.Module, critic_obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = all_action_critic_inputs(critic_obs).to(device)
    return critic_forward(critic, x).reshape(critic_obs.shape[0], N_ACTIONS, N_OBJECTIVES)

def rollout(actor, cfg: AdvantageTrapConfig, rng: np.random.Generator, device: torch.device):
    state, regime = initial_state(cfg.batch_episodes, rng)
    rows = []
    for t in range(cfg.horizon):
        actor_obs = actor_observe(state, t, cfg, rng)
        critic_obs = critic_observe(state, t, cfg, rng)
        weights = sample_weights(cfg.batch_episodes, cfg, rng)
        with torch.no_grad():
            probs = F.softmax(actor(actor_input(actor_obs, weights).to(device)), dim=1).cpu().numpy()
        actions = np.array([rng.choice(N_ACTIONS, p=p) for p in probs], dtype=np.int64)
        next_state, reward = step_env(state, regime, actions, t, cfg, rng, add_noise=True)
        if t < cfg.horizon - 1:
            next_actor_obs = actor_observe(next_state, t + 1, cfg, rng)
            next_critic_obs = critic_observe(next_state, t + 1, cfg, rng)
        else:
            next_actor_obs = np.zeros_like(actor_obs)
            next_critic_obs = np.zeros_like(critic_obs)
        rows.append((actor_obs, critic_obs, weights, actions, reward, next_actor_obs, next_critic_obs, t == cfg.horizon - 1))
        state = next_state
    return rows

def train_one(model_name: str, seed: int, cfg: AdvantageTrapConfig) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed + 9101)
    device = resolve_device(cfg.device)
    train_cfg = TrainConfig(hidden_dim=cfg.hidden_dim)
    actor = build_actor_policy(model_name, OBS_DIM + N_OBJECTIVES, OBS_DIM, N_ACTIONS, cfg.hidden_dim).to(device)
    critic = build_actor_critic_critic(model_name, OBS_DIM + N_ACTIONS, N_OBJECTIVES, train_cfg).to(device)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, weight_decay=1e-05)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=1e-05)
    curves = []
    train_advantage_vars = []
    actor_grad_norms = []
    critic_grad_norms = []
    for epoch in range(cfg.train_epochs):
        transitions = rollout(actor, cfg, rng, device)
        actor_obs = np.concatenate([x[0] for x in transitions], axis=0)
        critic_obs = np.concatenate([x[1] for x in transitions], axis=0)
        weights = np.concatenate([x[2] for x in transitions], axis=0)
        actions = np.concatenate([x[3] for x in transitions], axis=0)
        rewards = np.concatenate([x[4] for x in transitions], axis=0)
        next_actor_obs = np.concatenate([x[5] for x in transitions], axis=0)
        next_critic_obs = np.concatenate([x[6] for x in transitions], axis=0)
        done = np.concatenate([np.full(cfg.batch_episodes, x[7], dtype=np.float32) for x in transitions], axis=0)
        weights_t = torch.from_numpy(weights).float().to(device)
        rewards_t = torch.from_numpy(rewards).float().to(device)
        done_t = torch.from_numpy(done).float().to(device).unsqueeze(1)
        action_t = torch.from_numpy(actions).long().to(device)
        selected_x = critic_inputs(critic_obs, actions).to(device)
        pred = critic_forward(critic, selected_x)
        with torch.no_grad():
            next_probs = F.softmax(actor(actor_input(next_actor_obs, weights).to(device)), dim=1)
            next_q = critic_all_q(critic, next_critic_obs, device)
            next_v = (next_probs.unsqueeze(-1) * next_q).sum(dim=1)
            target = rewards_t + cfg.gamma * (1.0 - done_t) * next_v
        gates = critic.gate_weights(selected_x) if hasattr(critic, 'gate_weights') else None
        critic_loss = F.mse_loss(pred, target) + regularization_loss(gates, train_cfg).to(device)
        logits = actor(actor_input(actor_obs, weights).to(device))
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        selected_logp = log_probs.gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_all = critic_all_q(critic, critic_obs, device)
            scalar_q = (q_all * weights_t.unsqueeze(1)).sum(dim=2)
            baseline = (probs * scalar_q).sum(dim=1)
            scalar_target = (target * weights_t).sum(dim=1)
            advantage = scalar_target - baseline
            raw_advantage = advantage
            if cfg.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-06)
        train_advantage_vars.append(float(raw_advantage.var(unbiased=False).detach().cpu()))
        entropy = -(probs * torch.log(probs + 1e-08)).sum(dim=1).mean()
        actor_loss = -(selected_logp * advantage).mean() - cfg.entropy_weight * entropy
        critic_opt.zero_grad(set_to_none=True)
        actor_opt.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
        actor_grads = [p.grad.detach().flatten() for p in actor.parameters() if p.grad is not None]
        actor_grad_norms.append(float(torch.linalg.vector_norm(torch.cat(actor_grads)).cpu()) if actor_grads else np.nan)
        critic_grads = [p.grad.detach().flatten() for p in critic.parameters() if p.grad is not None]
        critic_grad_norms.append(float(torch.linalg.vector_norm(torch.cat(critic_grads)).cpu()) if critic_grads else np.nan)
        critic_opt.step()
        actor_opt.step()
        if (epoch + 1) % 20 == 0:
            curves.append({'epoch': epoch + 1, 'critic_loss': float(critic_loss.detach().cpu()), 'actor_loss': float(actor_loss.detach().cpu()), 'td_advantage_variance': train_advantage_vars[-1], 'actor_policy_gradient_norm': actor_grad_norms[-1], 'critic_gradient_norm': critic_grad_norms[-1]})
    metrics = evaluate(actor, critic, seed + 3329, cfg, device)
    grad_metrics = objective_gradient_diagnostics(actor, critic, seed + 5541, cfg, device)
    metrics.update({'variant': cfg.variant, 'model': model_name, 'seed': seed, 'horizon': cfg.horizon, 'train_epochs': cfg.train_epochs, 'parameter_count_actor': count_parameters(actor), 'parameter_count_critic': count_parameters(critic), 'train_td_advantage_variance_mean': float(np.mean(train_advantage_vars)), 'train_td_advantage_variance_last': float(train_advantage_vars[-1]), 'actor_policy_gradient_norm_mean': float(np.mean(actor_grad_norms)), 'actor_policy_gradient_norm_variance': float(np.var(actor_grad_norms)), 'actor_policy_gradient_norm_last': float(actor_grad_norms[-1]), 'critic_gradient_norm_mean': float(np.mean(critic_grad_norms)), 'critic_gradient_norm_variance': float(np.var(critic_grad_norms)), 'critic_gradient_norm_last': float(critic_grad_norms[-1])})
    metrics.update(grad_metrics)
    for row in curves:
        row.update({'variant': cfg.variant, 'model': model_name, 'seed': seed})
    return {'summary': metrics, 'curves': curves}

def oracle_actions(regime: np.ndarray, weights: np.ndarray, t: int) -> np.ndarray:
    actions = np.empty(regime.shape[0], dtype=np.int64)
    for r in range(3):
        idx = regime == r
        good = GOOD_ACTIONS[r]
        actions[idx] = good[t % len(good)]
    q2_focus = weights[:, 1] > 0.58
    q3_focus = weights[:, 2] > 0.58
    actions[(regime == 0) & q2_focus] = 1
    actions[(regime == 1) & q2_focus] = 3
    actions[(regime == 2) & q3_focus] = 6
    return actions

def objective_gradient_diagnostics(actor: torch.nn.Module, critic: torch.nn.Module, seed: int, cfg: AdvantageTrapConfig, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    transitions = rollout(actor, cfg, rng, device)
    actor_obs = np.concatenate([x[0] for x in transitions], axis=0)
    critic_obs = np.concatenate([x[1] for x in transitions], axis=0)
    weights = np.concatenate([x[2] for x in transitions], axis=0)
    actions = np.concatenate([x[3] for x in transitions], axis=0)
    rewards = np.concatenate([x[4] for x in transitions], axis=0)
    next_actor_obs = np.concatenate([x[5] for x in transitions], axis=0)
    next_critic_obs = np.concatenate([x[6] for x in transitions], axis=0)
    done = np.concatenate([np.full(cfg.batch_episodes, x[7], dtype=np.float32) for x in transitions], axis=0)
    action_t = torch.from_numpy(actions).long().to(device)
    rewards_t = torch.from_numpy(rewards).float().to(device)
    done_t = torch.from_numpy(done).float().to(device).unsqueeze(1)
    with torch.no_grad():
        next_probs = F.softmax(actor(actor_input(next_actor_obs, weights).to(device)), dim=1)
        next_q = critic_all_q(critic, next_critic_obs, device)
        next_v = (next_probs.unsqueeze(-1) * next_q).sum(dim=1)
        target = rewards_t + cfg.gamma * (1.0 - done_t) * next_v
        current_probs = F.softmax(actor(actor_input(actor_obs, weights).to(device)), dim=1)
        q_all = critic_all_q(critic, critic_obs, device)
    grads = []
    norms = []
    for objective_idx in range(N_OBJECTIVES):
        actor.zero_grad(set_to_none=True)
        logits = actor(actor_input(actor_obs, weights).to(device))
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        selected_logp = log_probs.gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            baseline = (current_probs * q_all[:, :, objective_idx]).sum(dim=1)
            objective_advantage = target[:, objective_idx] - baseline
        loss = -(selected_logp * objective_advantage).mean()
        loss.backward()
        flat = [p.grad.detach().flatten() for p in actor.parameters() if p.grad is not None]
        grad = torch.cat(flat) if flat else torch.zeros(1, device=device)
        grads.append(grad)
        norms.append(float(torch.linalg.vector_norm(grad).cpu()))
    cosines = []
    for i, j in itertools.combinations(range(N_OBJECTIVES), 2):
        denom = torch.linalg.vector_norm(grads[i]) * torch.linalg.vector_norm(grads[j]) + 1e-12
        cosines.append(float(torch.dot(grads[i], grads[j]).div(denom).cpu()))
    actor.zero_grad(set_to_none=True)
    return {'objective_gradient_cosine_mean': float(np.mean(cosines)), 'objective_gradient_cosine_min': float(np.min(cosines)), 'objective_gradient_conflict_ratio': float(np.mean(np.array(cosines) < 0.0)), 'objective_gradient_norm_variance': float(np.var(norms)), 'objective_gradient_norm_mean': float(np.mean(norms))}

def monte_carlo_action_returns(actor: torch.nn.Module, state: np.ndarray, regime: np.ndarray, weights: np.ndarray, cfg: AdvantageTrapConfig, device: torch.device) -> np.ndarray:
    returns = []
    for first_action in range(N_ACTIONS):
        sim_state = state.copy()
        total = np.zeros((state.shape[0], N_OBJECTIVES), dtype=np.float32)
        action_vec = np.full(state.shape[0], first_action, dtype=np.int64)
        for t in range(cfg.horizon):
            if t > 0:
                aobs = actor_observe(sim_state, t, cfg, np.random.default_rng(1000 + t))
                with torch.no_grad():
                    probs = F.softmax(actor(actor_input(aobs, weights).to(device)), dim=1)
                action_vec = probs.argmax(dim=1).cpu().numpy()
            sim_state, reward = step_env(sim_state, regime, action_vec, t, cfg, None, add_noise=False)
            total += reward
        returns.append(total)
    return np.stack(returns, axis=1)

def monte_carlo_policy_values(actor: torch.nn.Module, state: np.ndarray, regime: np.ndarray, weights: np.ndarray, cfg: AdvantageTrapConfig, device: torch.device, start_t: int) -> np.ndarray:
    sim_state = state.copy()
    total = np.zeros((state.shape[0], N_OBJECTIVES), dtype=np.float32)
    for t in range(start_t, cfg.horizon):
        aobs = actor_observe(sim_state, t, cfg, np.random.default_rng(2000 + t))
        with torch.no_grad():
            probs = F.softmax(actor(actor_input(aobs, weights).to(device)), dim=1)
        actions = probs.argmax(dim=1).cpu().numpy()
        sim_state, reward = step_env(sim_state, regime, actions, t, cfg, None, add_noise=False)
        total += reward
    return total

def evaluate(actor, critic, seed: int, cfg: AdvantageTrapConfig, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for eval_name, w in EVAL_WEIGHTS.items():
        state, regime = initial_state(cfg.eval_episodes, rng)
        oracle_state = state.copy()
        total = np.zeros((cfg.eval_episodes, 3), dtype=np.float32)
        oracle_total = np.zeros((cfg.eval_episodes, 3), dtype=np.float32)
        match, entropy, critic_errors, adv_errors = ([], [], [], [])
        eval_w = np.tile(w, (cfg.eval_episodes, 1)).astype(np.float32)
        for t in range(cfg.horizon):
            aobs = actor_observe(state, t, cfg, rng)
            cobs = critic_observe(state, t, cfg, None)
            with torch.no_grad():
                logits = actor(actor_input(aobs, eval_w).to(device))
                probs = F.softmax(logits, dim=1)
                actions = probs.argmax(dim=1).cpu().numpy()
                q_pred = critic_all_q(critic, cobs, device).cpu().numpy()
            next_state, reward = step_env(state, regime, actions, t, cfg, None, add_noise=False)
            total += reward
            oa = oracle_actions(regime, eval_w, t)
            oracle_next, oracle_reward = step_env(oracle_state, regime, oa, t, cfg, None, add_noise=False)
            oracle_total += oracle_reward
            immediate = []
            for a in range(N_ACTIONS):
                _, r = step_env(state.copy(), regime, np.full(cfg.eval_episodes, a, dtype=np.int64), t, cfg, None, add_noise=False)
                immediate.append(r)
            immediate = np.stack(immediate, axis=1)
            pred_scalar = np.einsum('naj,nj->na', q_pred, eval_w)
            true_scalar = np.einsum('naj,nj->na', immediate, eval_w)
            probs_np = probs.cpu().numpy()
            pred_adv = pred_scalar - (probs_np * pred_scalar).sum(axis=1, keepdims=True)
            true_adv = true_scalar - (probs_np * true_scalar).sum(axis=1, keepdims=True)
            adv_error = (pred_adv - true_adv).reshape(-1)
            critic_errors.append(((q_pred - immediate) ** 2).mean())
            adv_errors.append(adv_error)
            match.append((actions == oa).mean())
            entropy.append((-(probs_np * np.log(probs_np + 1e-08)).sum(axis=1)).mean())
            state, oracle_state = (next_state, oracle_next)
        scalar_total = total @ w
        scalar_oracle = oracle_total @ w
        prefix = f'{eval_name}_'
        out[prefix + 'return_mean'] = float(scalar_total.mean())
        out[prefix + 'oracle_return_mean'] = float(scalar_oracle.mean())
        out[prefix + 'regret'] = float((scalar_oracle - scalar_total).mean())
        out[prefix + 'action_match'] = float(np.mean(match))
        out[prefix + 'policy_entropy'] = float(np.mean(entropy))
        out[prefix + 'q1_return'] = float(total[:, 0].mean())
        out[prefix + 'q2_return'] = float(total[:, 1].mean())
        out[prefix + 'q3_return'] = float(total[:, 2].mean())
        out[prefix + 'critic_immediate_mse'] = float(np.mean(critic_errors))
        all_adv_errors = np.concatenate(adv_errors)
        out[prefix + 'advantage_error_variance'] = float(np.var(all_adv_errors))
        out[prefix + 'advantage_error_mae'] = float(np.mean(np.abs(all_adv_errors)))
        mc_rng = np.random.default_rng(seed + 7000 + len(out))
        mc_state, mc_regime = initial_state(min(cfg.mc_eval_episodes, cfg.eval_episodes), mc_rng)
        mc_weights = np.tile(w, (mc_state.shape[0], 1)).astype(np.float32)
        mc_actor_obs = actor_observe(mc_state, 0, cfg, mc_rng)
        mc_critic_obs = critic_observe(mc_state, 0, cfg, None)
        with torch.no_grad():
            mc_probs = F.softmax(actor(actor_input(mc_actor_obs, mc_weights).to(device)), dim=1).cpu().numpy()
            mc_q_pred = critic_all_q(critic, mc_critic_obs, device).cpu().numpy()
        mc_returns = monte_carlo_action_returns(actor, mc_state, mc_regime, mc_weights, cfg, device)
        mc_pred_scalar = np.einsum('naj,nj->na', mc_q_pred, mc_weights)
        mc_true_scalar = np.einsum('naj,nj->na', mc_returns, mc_weights)
        mc_pred_adv = mc_pred_scalar - (mc_probs * mc_pred_scalar).sum(axis=1, keepdims=True)
        mc_true_adv = mc_true_scalar - (mc_probs * mc_true_scalar).sum(axis=1, keepdims=True)
        mc_adv_error = (mc_pred_adv - mc_true_adv).reshape(-1)
        out[prefix + 'mc_advantage_error_mse'] = float(np.mean(mc_adv_error ** 2))
        out[prefix + 'mc_advantage_error_variance'] = float(np.var(mc_adv_error))
        out[prefix + 'mc_advantage_error_mae'] = float(np.mean(np.abs(mc_adv_error)))
        td_pred_advantages = []
        td_true_advantages = []
        for first_action in range(N_ACTIONS):
            action_vec = np.full(mc_state.shape[0], first_action, dtype=np.int64)
            next_state, reward = step_env(mc_state.copy(), mc_regime, action_vec, 0, cfg, None, add_noise=False)
            next_critic_obs = critic_observe(next_state, 1, cfg, None)
            with torch.no_grad():
                next_actor_obs = actor_observe(next_state, 1, cfg, mc_rng)
                next_probs = F.softmax(actor(actor_input(next_actor_obs, mc_weights).to(device)), dim=1).cpu().numpy()
                next_q_pred = critic_all_q(critic, next_critic_obs, device).cpu().numpy()
            next_pred_scalar = np.einsum('naj,nj->na', next_q_pred, mc_weights)
            pred_next_value = (next_probs * next_pred_scalar).sum(axis=1)
            pred_current_value = (mc_probs * mc_pred_scalar).sum(axis=1)
            reward_scalar = np.einsum('nj,nj->n', reward, mc_weights)
            td_pred_advantages.append(reward_scalar + cfg.gamma * pred_next_value - pred_current_value)
            next_true_value = monte_carlo_policy_values(actor, next_state, mc_regime, mc_weights, cfg, device, start_t=1)
            next_true_scalar = np.einsum('nj,nj->n', next_true_value, mc_weights)
            true_current_value = (mc_probs * mc_true_scalar).sum(axis=1)
            td_true_advantages.append(reward_scalar + cfg.gamma * next_true_scalar - true_current_value)
        td_pred_adv = np.stack(td_pred_advantages, axis=1)
        td_true_adv = np.stack(td_true_advantages, axis=1)
        td_adv_error = (td_pred_adv - td_true_adv).reshape(-1)
        out[prefix + 'td_advantage_error_mse'] = float(np.mean(td_adv_error ** 2))
        out[prefix + 'td_advantage_error_variance'] = float(np.var(td_adv_error))
        out[prefix + 'td_advantage_error_mae'] = float(np.mean(np.abs(td_adv_error)))
    return out

def add_random_average_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in ('return_mean', 'regret', 'action_match', 'policy_entropy', 'critic_immediate_mse', 'advantage_error_variance', 'advantage_error_mae', 'mc_advantage_error_mse', 'mc_advantage_error_variance', 'mc_advantage_error_mae', 'td_advantage_error_mse', 'td_advantage_error_variance', 'td_advantage_error_mae', 'q1_return', 'q2_return', 'q3_return'):
        columns = [f'{prefix}_{metric}' for prefix in PREFIXES]
        if all((column in out.columns for column in columns)):
            out[f'random_avg_{metric}'] = out[columns].mean(axis=1)
    return out
