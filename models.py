"""Section 4 dependencies only; extracted from the project source. See SOURCE_MAP.json."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticFC(nn.Module):

    def __init__(self, input_dim: int, num_objectives: int, hidden_dim: int=126, depth: int=3):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden_dim), nn.ReLU()])
            last = hidden_dim
        layers.append(nn.Linear(last, num_objectives))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CriticMultiHead(nn.Module):

    def __init__(self, input_dim: int, num_objectives: int, hidden_dim: int=176):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_objectives)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.cat([head(z) for head in self.heads], dim=1)

class CriticRegimeMoE(nn.Module):

    def __init__(self, input_dim: int, num_objectives: int, hidden_dim: int=96, n_experts: int=3, n_actions: int=8, gate_temperature: float=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.gate_input_dim = input_dim - n_actions
        self.gate_temperature = gate_temperature
        self.gate = nn.Sequential(nn.Linear(self.gate_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_objectives)) for _ in range(n_experts)])

    def gate_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.gate_input_dim]

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(self.gate_inputs(x)) / self.gate_temperature, dim=1)

    def forward(self, x: torch.Tensor, return_gate: bool=False):
        weights = self.gate_weights(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        pred = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return (pred, weights) if return_gate else pred

class CriticRegimeMHMoE(nn.Module):

    def __init__(self, input_dim: int, num_objectives: int, hidden_dim: int=96, n_experts: int=3, n_actions: int=8, gate_temperature: float=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.gate_input_dim = input_dim - n_actions
        self.gate_temperature = gate_temperature
        self.gate = nn.Sequential(nn.Linear(self.gate_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_experts)])
        self.heads = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_objectives)]) for _ in range(n_experts)])

    def gate_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.gate_input_dim]

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(self.gate_inputs(x)) / self.gate_temperature, dim=1)

    def forward(self, x: torch.Tensor, return_gate: bool=False):
        weights = self.gate_weights(x)
        expert_outputs = []
        for expert, heads in zip(self.experts, self.heads):
            z = expert(x)
            expert_outputs.append(torch.cat([head(z) for head in heads], dim=1))
        stacked = torch.stack(expert_outputs, dim=1)
        pred = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return (pred, weights) if return_gate else pred

def count_parameters(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def build_model(name: str, input_dim: int, num_objectives: int, hidden_dim: int=96, n_experts: int=3) -> nn.Module:
    if name == 'FC':
        return CriticFC(input_dim, num_objectives, hidden_dim=125, depth=3)
    if name == 'MH':
        return CriticMultiHead(input_dim, num_objectives, hidden_dim=174)
    if name == 'MoE':
        return CriticRegimeMoE(input_dim, num_objectives, hidden_dim, n_experts)
    if name == 'MH+MoE':
        return CriticRegimeMHMoE(input_dim, num_objectives, hidden_dim, n_experts)
    raise ValueError(f'Unknown model: {name}')
