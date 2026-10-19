"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainConfig
from models import build_model

class ActorPolicy(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class GeneralActorMoE(nn.Module):

    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int=64, n_experts: int=3):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_actions)) for _ in range(n_experts)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        gates = F.softmax(self.gate(obs), dim=1)
        expert_logits = torch.stack([expert(obs) for expert in self.experts], dim=1)
        return (gates.unsqueeze(-1) * expert_logits).sum(dim=1)

def build_actor_critic_critic(model_name: str, input_dim: int, num_objectives: int, train_cfg: TrainConfig) -> nn.Module:
    return build_model(critic_name_for_model(model_name), input_dim, num_objectives, train_cfg.hidden_dim, train_cfg.n_experts)

def critic_name_for_model(model_name: str) -> str:
    if model_name == 'ActorMoE':
        return 'FC'
    return model_name

def build_actor_policy(model_name: str, actor_dim: int, obs_dim: int, n_actions: int, hidden_dim: int) -> nn.Module:
    if model_name == 'ActorMoE':
        return GeneralActorMoE(actor_dim, n_actions, hidden_dim=50)
    return ActorPolicy(actor_dim, n_actions, hidden_dim)

def critic_forward(critic: nn.Module, x: torch.Tensor, weights: torch.Tensor | None=None, return_gate: bool=False):
    if return_gate and hasattr(critic, 'gate_weights'):
        return critic(x, return_gate=True)
    return critic(x)
