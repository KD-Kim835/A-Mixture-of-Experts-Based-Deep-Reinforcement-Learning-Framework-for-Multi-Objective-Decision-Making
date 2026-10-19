"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import itertools
import numpy as np
import torch
import torch.nn.functional as F

def gradient_interference(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, max_samples: int=512) -> dict[str, float]:
    model.eval()
    xb = x[:max_samples]
    yb = y[:max_samples]
    grads = []
    for obj in range(yb.shape[1]):
        model.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred[:, obj], yb[:, obj])
        loss.backward()
        flat = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
        grads.append(torch.cat(flat))
    cosines = [F.cosine_similarity(grads[i], grads[j], dim=0).item() for i, j in itertools.combinations(range(len(grads)), 2)]
    arr = np.array(cosines, dtype=np.float32)
    norms = np.array([float(torch.linalg.vector_norm(g).detach().cpu()) for g in grads], dtype=np.float32)
    negative_cosines = np.maximum(0.0, -arr)
    return {'grad_cosine_mean': float(arr.mean()), 'grad_cosine_min': float(arr.min()), 'gradient_conflict_ratio': float((arr < 0.0).mean()), 'gradient_negative_cosine_magnitude': float(negative_cosines.mean()), 'gradient_negative_cosine_max': float(negative_cosines.max()), 'gradient_norm_mean': float(norms.mean()), 'gradient_norm_std': float(norms.std()), 'gradient_norm_variance': float(norms.var()), 'gradient_norm_min': float(norms.min()), 'gradient_norm_max': float(norms.max()), 'gradient_norm_cv': float(norms.std() / (norms.mean() + 1e-08))}

def best_gate_regime_alignment(gates: np.ndarray, ctx_true: np.ndarray) -> tuple[float, dict[int, int]]:
    hard = gates.argmax(axis=1)
    n_experts = gates.shape[1]
    best_acc = -1.0
    best_map: dict[int, int] = {}
    for perm in itertools.permutations(range(3), min(3, n_experts)):
        mapping = {expert: regime for expert, regime in enumerate(perm)}
        pred = np.array([mapping.get(e, -1) for e in hard])
        acc = float((pred == ctx_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_map = mapping
    return (best_acc, best_map)

@torch.no_grad()
def gate_diagnostics(model: torch.nn.Module, x: torch.Tensor, ctx_true: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not hasattr(model, 'gate_weights'):
        return ({'gate_entropy': np.nan, 'effective_experts': np.nan, 'gate_regime_alignment': np.nan}, [])
    weights = model.gate_weights(x).detach().cpu().numpy()
    eps = 1e-08
    entropy = -(weights * np.log(weights + eps)).sum(axis=1)
    effective = np.exp(entropy)
    alignment, _ = best_gate_regime_alignment(weights, ctx_true)
    summary = {'gate_entropy': float(entropy.mean()), 'effective_experts': float(effective.mean()), 'gate_regime_alignment': alignment}
    rows = []
    for regime in sorted(np.unique(ctx_true)):
        mask = ctx_true == regime
        means = weights[mask].mean(axis=0)
        row = {'ctx_true': int(regime)}
        row.update({f'expert_{i}': float(v) for i, v in enumerate(means)})
        row['gate_entropy'] = float(entropy[mask].mean())
        row['effective_experts'] = float(effective[mask].mean())
        rows.append(row)
    return (summary, rows)
