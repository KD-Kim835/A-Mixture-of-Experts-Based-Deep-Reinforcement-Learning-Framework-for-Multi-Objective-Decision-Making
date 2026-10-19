"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import TrainConfig

def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def regularization_loss(gates: torch.Tensor | None, cfg: TrainConfig) -> torch.Tensor:
    if gates is None:
        return torch.tensor(0.0)
    eps = 1e-08
    entropy = -(gates * torch.log(gates + eps)).sum(dim=1).mean()
    mean_gate = gates.mean(dim=0)
    uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
    load_balance = F.mse_loss(mean_gate, uniform)
    return -cfg.gate_entropy_weight * entropy + cfg.load_balance_weight * load_balance

def fit_model(model: torch.nn.Module, train_data: dict, test_data: dict, cfg: TrainConfig, seed: int) -> dict:
    set_seed(seed)
    device = resolve_device(cfg.device)
    model.to(device)
    train_x = train_data['X'].to(device)
    train_y = train_data['Q'].to(device)
    test_x = test_data['X'].to(device)
    test_y = test_data['Q'].to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_curve = []
    start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            gates = None
            if hasattr(model, 'gate_weights'):
                pred, gates = model(xb, return_gate=True)
            else:
                pred = model(xb)
            mse = F.mse_loss(pred, yb)
            loss = mse + regularization_loss(gates, cfg).to(device)
            loss.backward()
            opt.step()
            epoch_losses.append(float(mse.detach().cpu()))
        train_curve.append(float(np.mean(epoch_losses)))
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        mse_by_obj = ((pred - test_y) ** 2).mean(dim=0).detach().cpu().numpy()
        total_mse = float(mse_by_obj.mean())
    return {'model': model, 'train_curve': train_curve, 'training_time_sec': elapsed, 'test_mse': total_mse, 'test_mse_by_objective': mse_by_obj}
