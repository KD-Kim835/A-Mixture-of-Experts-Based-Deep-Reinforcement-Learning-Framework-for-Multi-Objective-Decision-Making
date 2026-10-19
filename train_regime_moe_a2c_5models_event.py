"""
Five-model event-triggered A2C benchmark with unified basic actor/critic losses.

Compared models
---------------
1. FC: shared station-wise actor + fully connected scalar critic.
2. MH: shared station-wise actor + five objective-specific critic heads.
3. MoE: shared station-wise actor + four regime experts, four regime-specific scalar value heads, and learned gate-weighted value aggregation.
4. MH+MoE: shared station-wise actor + MH-style five objective heads, objective fusion FC, four regime-specific value heads, and learned gate-weighted value aggregation.
5. Actor-MoE: residual regime-gated actor corrections + fully connected scalar critic.

Basic-loss comparison rule
--------------------------
* Every architecture uses the same one-step SMDP TD error
  delta = r_interval + gamma**duration * V(s_next) - V(s).
* Every critic minimizes MSE(V(s), TD target).
* Every actor minimizes the same negative TD-error-weighted log likelihood,
  with the same entropy regularization.
* MoE gates retain only supervised regime cross-entropy as a separate gate loss.
* No GAE, objective-return loss, scalar-consistency loss, expert-TD loss,
  diversity loss, gate-balance loss, or gate-entropy loss enters optimization.

Key corrections
---------------
* Four 3-way station actions replace one 81-way joint categorical actor.
* Actor log probabilities and entropy use only active/loadable stations.
* Event transitions use discounted rewards inside each variable-duration interval
  and gamma**duration bootstrapping in SMDP-aware GAE.
* Stock-removal feasibility is enforced by station-wise action masks.
* Gate training uses genuine exogenous regimes, stronger supervised routing,
  true-regime expert losses, empirical-balance regularization, and conditional
  temperature annealing. The v6 environment creates distinct regime-specific
  station-allocation and throughput-quality trade-offs.
* Every model is parameter-budgeted to approximately 10,000 trainable parameters.
* MoE performs gating after four independent regime-specific scalar value heads rather than mixing expert features before a shared head.
* MH+MoE preserves the MH objective-decomposition mechanism first, then adds an objective-fusion FC layer, four regime-specific scalar value heads, and gate-weighted value aggregation.
* All five models are parameter-budgeted to approximately 10,000 trainable parameters.
* Training and validation figures are generated separately for every replication.
* After the final replication, the same manuscript-style figures are aggregated as
  mean ± standard deviation across replications.
* Raw SMDP-GAE, normalized GAE, and active-station advantage diagnostics are
  recorded for every model at every training episode and validation update.
* Oracle-routing validation compares learned versus true-regime routing without
  leaking oracle labels into ordinary deployment evaluation.
* Actor-MoE v7 uses a generalist base actor, residual regime experts, a separate
  gate encoder, and an actual oracle-to-blended-to-learned behavior curriculum.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from manufacturing_queue_regime_env_event_v8_equal_reward_weights import (
    ManufacturingQueueRegimeEventEnv,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

# ---------------------------------------------------------------------
# Improved learning defaults
# ---------------------------------------------------------------------
ACTOR_LR = 1.0e-4
CRITIC_LR = 2.0e-4
GATE_LR = 1.0e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
ENTROPY_START = 0.012
ENTROPY_END = 0.002
ROLLOUT_DECISIONS = 128
ACTOR_GRAD_CLIP = 2.0
CRITIC_GRAD_CLIP = 5.0
GATE_GRAD_CLIP = 2.0

# Stabilization and credit-assignment defaults.
CRITIC_REWARD_SCALE = 0.10
STATION_DIFFERENCE_CREDIT_COEF = 0.0  # disabled: reward-only actor credit
SETPOINT_GUIDANCE_COEF = 0.0  # disabled: no auxiliary policy shaping
SETPOINT_GUIDANCE_END_PROGRESS = 0.25
ORACLE_ROUTING_END_PROGRESS = 0.30
BLENDED_ROUTING_END_PROGRESS = 0.70
GATE_ALIGNMENT_SHARPEN_THRESHOLD = 0.60
EXPERT_DIVERSITY_COEF = 0.003
ACTOR_RESIDUAL_LR_MULTIPLIER = 1.50

TARGET_TRAINABLE_PARAMETERS = 10_000
# Architectures are balanced around 10,000 total trainable parameters.
PARAMETER_TOLERANCE = 50
REPLICATION_ROLLING_WINDOW = 20

STOCK_TARGET_TENSOR = torch.tensor(
    [1.50, 0.90, 0.45, 0.25], dtype=torch.float32, device=device
)
TARGET_TOTAL_STOCK = float(STOCK_TARGET_TENSOR.sum().item())
DIMENSIONAL_TOLERANCE = 0.08 * TARGET_TOTAL_STOCK

NUM_STATIONS = 4
NUM_STATION_ACTIONS = 3
REGIME_NAMES = ManufacturingQueueRegimeEventEnv.REGIME_NAMES
NUM_REGIMES = len(REGIME_NAMES)
OBJECTIVE_NAMES = [
    "step_yield",
    "step_type1",
    "step_type2",
    "step_buffer_change",
    "step_lost_orders",
]
NUM_OBJECTIVES = len(OBJECTIVE_NAMES)
# One-to-one reward decomposition: dY - dD_I - dD_II - db - dL.
OBJECTIVE_AGGREGATION_WEIGHTS = (1.0, -1.0, -1.0, -1.0, -1.0)
OBJECTIVE_AGGREGATION_WEIGHTS_TENSOR = torch.tensor(
    OBJECTIVE_AGGREGATION_WEIGHTS, dtype=torch.float32, device=device
)

MODEL_DISPLAY_NAMES = {
    "fc": "FC",
    "regime_heads": "MH",
    "critic_moe": "MoE",
    "multihead_critic_moe": "MH + MoE",
    "actor_moe": "Actor-MoE",
}
MODEL_ORDER = [
    "fc",
    "regime_heads",
    "critic_moe",
    "multihead_critic_moe",
    "actor_moe",
]
GATED_MODELS = ["critic_moe", "multihead_critic_moe", "actor_moe"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_label(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_parameter_iterable(parameters: Iterable[nn.Parameter]) -> int:
    """Count unique trainable parameters from an iterable."""
    seen = set()
    total = 0
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def parameter_breakdown(model: "BaseA2C") -> Dict[str, int]:
    """Return actor/critic/total counts and verify that groups do not overlap."""
    actor_parameters = list(model.actor_parameters())
    critic_parameters = list(model.critic_parameters())
    actor_ids = {id(parameter) for parameter in actor_parameters if parameter.requires_grad}
    critic_ids = {id(parameter) for parameter in critic_parameters if parameter.requires_grad}
    overlap = actor_ids.intersection(critic_ids)
    if overlap:
        raise RuntimeError("Actor and critic optimizer groups contain shared parameters.")
    actor_count = count_parameter_iterable(actor_parameters)
    critic_count = count_parameter_iterable(critic_parameters)
    total_count = count_parameters(model)
    if actor_count + critic_count != total_count:
        raise RuntimeError(
            "Actor/critic parameter groups do not cover all trainable parameters: "
            f"actor={actor_count}, critic={critic_count}, total={total_count}."
        )
    return {
        "actor_trainable_parameters": actor_count,
        "critic_trainable_parameters": critic_count,
        "num_trainable_parameters": total_count,
        "parameter_difference_from_target": total_count - TARGET_TRAINABLE_PARAMETERS,
    }


def validate_parameter_budget(model_name: str, breakdown: Dict[str, int]) -> None:
    difference = abs(int(breakdown["parameter_difference_from_target"]))
    if difference > PARAMETER_TOLERANCE:
        raise RuntimeError(
            f"{model_label(model_name)} has {breakdown['num_trainable_parameters']:,} "
            f"trainable parameters; target={TARGET_TRAINABLE_PARAMETERS:,} "
            f"with tolerance=±{PARAMETER_TOLERANCE}."
        )


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = float(parameter.grad.detach().norm(2).item())
            total_sq += value * value
    return float(total_sq**0.5)


def mlp(
    in_dim: int,
    hidden: Tuple[int, ...],
    out_dim: int,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = int(in_dim)
    for width in hidden:
        layers.append(nn.Linear(previous, int(width)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(nn.ReLU())
        previous = int(width)
    layers.append(nn.Linear(previous, int(out_dim)))
    return nn.Sequential(*layers)


def discounted_interval_values(
    primitive_values: np.ndarray,
    discount: float = GAMMA,
) -> np.ndarray:
    """Discount primitive rewards/components within one event interval."""
    values = np.asarray(primitive_values, dtype=np.float32)
    if values.size == 0:
        return np.asarray(0.0, dtype=np.float32)
    weights = np.power(discount, np.arange(values.shape[0], dtype=np.float32))
    if values.ndim == 1:
        return np.asarray(np.dot(weights, values), dtype=np.float32)
    reshape = (values.shape[0],) + (1,) * (values.ndim - 1)
    return np.sum(values * weights.reshape(reshape), axis=0).astype(np.float32)


def _present_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-square-root weights for classes present in the current rollout."""
    counts = torch.bincount(targets.detach(), minlength=num_classes).float()
    weights = torch.zeros(num_classes, device=targets.device)
    present = counts > 0
    if present.any():
        inverse = torch.rsqrt(counts[present].clamp_min(1.0))
        weights[present] = inverse / inverse.mean().clamp_min(1e-8)
    else:
        weights.fill_(1.0)
    return weights


def balanced_gate_cross_entropy(
    logits: torch.Tensor,
    regime_ids: torch.Tensor,
    num_regimes: int,
    label_smoothing: float = 0.03,
) -> torch.Tensor:
    weights = _present_class_weights(regime_ids, num_regimes)
    return F.cross_entropy(
        logits,
        regime_ids,
        weight=weights,
        label_smoothing=label_smoothing,
    )


def balanced_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    regime_ids: torch.Tensor,
    num_regimes: int,
) -> torch.Tensor:
    """Regime-balanced Smooth L1 loss for scalar or vector predictions."""
    per_element = F.smooth_l1_loss(
        predictions,
        targets.detach(),
        reduction="none",
    )
    per_sample = per_element.reshape(per_element.size(0), -1).mean(dim=1)
    class_weights = _present_class_weights(regime_ids, num_regimes)
    sample_weights = class_weights[regime_ids]
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def gate_losses_and_metrics(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    regime_ids: torch.Tensor,
    num_regimes: int,
) -> Dict[str, torch.Tensor]:
    batch_index = torch.arange(regime_ids.size(0), device=regime_ids.device)
    # Cross-entropy receives untempered logits so its scale is not changed by
    # the routing-temperature schedule. Probabilities remain temperature-scaled.
    gate_ce = balanced_gate_cross_entropy(logits, regime_ids, num_regimes)
    gate_entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=1).mean()

    empirical = torch.bincount(regime_ids, minlength=num_regimes).float()
    empirical = empirical / empirical.sum().clamp_min(1.0)
    average_gate = probabilities.mean(dim=0)
    # Match utilization to the labels observed in the rollout instead of
    # forcing a uniform gate when the underlying regime distribution is not uniform.
    balance_loss = F.kl_div(
        torch.log(average_gate.clamp_min(1e-8)),
        empirical,
        reduction="sum",
    )
    gate_acc = (probabilities.argmax(dim=1) == regime_ids).float().mean()
    alignment = probabilities[batch_index, regime_ids].mean()
    return {
        "gate_ce": gate_ce,
        "gate_entropy_tensor": gate_entropy,
        "gate_balance": balance_loss,
        "gate_acc_tensor": gate_acc,
        "gate_alignment_tensor": alignment,
    }


# ---------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------
class RolloutMixin:
    def put_data(self, transition) -> None:
        self.data.append(transition)

    def make_batch(self, clear: bool = True):
        states, actions, rewards, reward_vectors, next_states = [], [], [], [], []
        continuations, valid_masks, active_masks, durations = [], [], [], []
        station_credits, regime_ids, next_regime_ids = [], [], []
        for transition in self.data:
            (
                state,
                action,
                reward,
                reward_vector,
                next_state,
                truncated,
                done,
                valid_action_mask,
                active_station_mask,
                elapsed_ticks,
                station_difference_reward,
                regime_id,
                next_regime_id,
            ) = transition
            states.append(state)
            actions.append(action)
            rewards.append([reward])
            reward_vectors.append(reward_vector)
            next_states.append(next_state)
            continuations.append([0.0 if (done or truncated) else 1.0])
            valid_masks.append(valid_action_mask)
            active_masks.append(active_station_mask)
            durations.append([elapsed_ticks])
            station_credits.append(station_difference_reward)
            regime_ids.append(regime_id)
            next_regime_ids.append(next_regime_id)
        if clear:
            self.data = []
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(actions), dtype=torch.long, device=device),
            torch.tensor(np.asarray(rewards), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(reward_vectors), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(continuations), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(valid_masks), dtype=torch.bool, device=device),
            torch.tensor(np.asarray(active_masks), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(durations), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(station_credits), dtype=torch.float32, device=device),
            torch.tensor(regime_ids, dtype=torch.long, device=device),
            torch.tensor(next_regime_ids, dtype=torch.long, device=device),
        )


# ---------------------------------------------------------------------
# Actor networks
# ---------------------------------------------------------------------
class FactorizedActorNetwork(nn.Module):
    """One shared actor with four station-wise 3-action output heads."""

    def __init__(
        self,
        state_dim: int,
        hidden: Tuple[int, int] = (42, 32),
    ) -> None:
        super().__init__()
        self.net = mlp(
            state_dim,
            hidden,
            NUM_STATIONS * NUM_STATION_ACTIONS,
            layer_norm=True,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.net(states)
        return logits.view(-1, NUM_STATIONS, NUM_STATION_ACTIONS)


class ActorMoENetwork(nn.Module):
    """Residual actor MoE with separate policy and gate encoders.

    A generalist station-wise base actor represents behavior shared by all regimes.
    Each expert learns only a bounded logit correction.  The routing weights are
    detached from the policy-gradient path, so the gate is trained only by explicit
    regime supervision rather than by noisy policy gradients.
    """

    def __init__(
        self,
        state_dim: int,
        num_experts: int = NUM_REGIMES,
        gate_temp_start: float = 1.5,
        gate_temp_end: float = 0.6,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.gate_temp_start = float(gate_temp_start)
        self.gate_temp_end = float(gate_temp_end)
        self.gate_temperature = self.gate_temp_start
        self.residual_scale = float(residual_scale)

        # Policy path: common representation + stable base policy + small residuals.
        # Parameter count is chosen so Actor-MoE remains within the 100k total budget.
        self.policy_encoder = nn.Sequential(
            nn.Linear(state_dim, 23),
            nn.LayerNorm(23),
            nn.ReLU(),
            nn.Linear(23, 22),
            nn.ReLU(),
        )
        self.base_head = nn.Linear(
            22, NUM_STATIONS * NUM_STATION_ACTIONS
        )
        self.residual_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(22, 5),
                    nn.ReLU(),
                    nn.Linear(5, NUM_STATIONS * NUM_STATION_ACTIONS),
                )
                for _ in range(self.num_experts)
            ]
        )

        # Gate path is completely separate from the policy representation.
        self.gate_encoder = nn.Sequential(
            nn.Linear(state_dim, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(10, self.num_experts)

        # Start residual corrections close to zero so the initial policy behaves
        # like the stable generalist base actor instead of a random expert mixture.
        for expert in self.residual_experts:
            final_layer = expert[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def policy_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.policy_encoder.parameters())
            + list(self.base_head.parameters())
            + list(self.residual_experts.parameters())
        )

    def gate_parameters(self) -> List[nn.Parameter]:
        return list(self.gate_encoder.parameters()) + list(
            self.gate_head.parameters()
        )

    def set_temperature(self, progress: float) -> None:
        p = float(np.clip(progress, 0.0, 1.0))
        self.gate_temperature = self.gate_temp_start + p * (
            self.gate_temp_end - self.gate_temp_start
        )

    def gate_outputs(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_latent = self.gate_encoder(states)
        gate_logits = self.gate_head(gate_latent)
        gate_probabilities = F.softmax(
            gate_logits / max(self.gate_temperature, 1e-6), dim=-1
        )
        return gate_probabilities, gate_logits

    def _routing_weights(
        self,
        learned_probabilities: torch.Tensor,
        regime_ids: Optional[torch.Tensor],
        oracle_weight: float,
        routing_mode: str,
    ) -> torch.Tensor:
        mode = str(routing_mode).lower()
        if mode not in {"learned", "oracle", "curriculum"}:
            raise ValueError(f"Unknown Actor-MoE routing_mode={routing_mode}")

        learned = learned_probabilities.detach()
        if mode == "learned" or regime_ids is None:
            return learned

        oracle = F.one_hot(
            regime_ids, num_classes=self.num_experts
        ).to(dtype=learned.dtype)
        if mode == "oracle":
            return oracle

        weight = float(np.clip(oracle_weight, 0.0, 1.0))
        routing = weight * oracle + (1.0 - weight) * learned
        return routing / routing.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        states: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        oracle_weight: float = 0.0,
        routing_mode: str = "learned",
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        policy_latent = self.policy_encoder(states)
        base_logits = self.base_head(policy_latent).view(
            -1, NUM_STATIONS, NUM_STATION_ACTIONS
        )
        residual_logits = torch.stack(
            [
                expert(policy_latent).view(
                    -1, NUM_STATIONS, NUM_STATION_ACTIONS
                )
                for expert in self.residual_experts
            ],
            dim=1,
        )  # [B, E, S, A]

        gate_probabilities, gate_logits = self.gate_outputs(states)
        routing_weights = self._routing_weights(
            gate_probabilities,
            regime_ids,
            oracle_weight=oracle_weight,
            routing_mode=routing_mode,
        )

        # Bounded residuals prevent one poorly trained expert from destroying the
        # generalist policy.  Confidence controls how strongly experts are used.
        bounded_residuals = torch.tanh(residual_logits)
        routed_residual = (
            bounded_residuals
            * routing_weights[:, :, None, None]
        ).sum(dim=1)
        route_confidence = routing_weights.max(dim=1).values
        uniform_probability = 1.0 / float(self.num_experts)
        residual_alpha = (
            (route_confidence - uniform_probability)
            / max(1e-8, 1.0 - uniform_probability)
        ).clamp(0.0, 1.0)
        combined_logits = (
            base_logits
            + self.residual_scale
            * residual_alpha[:, None, None]
            * routed_residual
        )
        combined_logits = combined_logits.masked_fill(
            ~valid_action_mask, float("-inf")
        )
        probabilities = F.softmax(combined_logits, dim=-1)
        return (
            probabilities,
            gate_probabilities,
            residual_logits,
            gate_logits,
            residual_alpha,
            routing_weights,
        )

    def oracle_probabilities(
        self,
        states: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Use the base policy plus the true-regime residual expert."""
        return self.forward(
            states,
            valid_action_mask,
            regime_ids=regime_ids,
            oracle_weight=1.0,
            routing_mode="oracle",
        )[0]


# ---------------------------------------------------------------------
# Base A2C
# ---------------------------------------------------------------------
class BaseA2C(nn.Module, RolloutMixin):
    gate_scope = "none"

    def standard_policy(
        self,
        states: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.actor(states)
        logits = logits.masked_fill(~valid_action_mask, float("-inf"))
        return F.softmax(logits, dim=-1)

    def policy_outputs(
        self,
        states: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ):
        return self.standard_policy(states, valid_action_mask), None

    def select_action(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        stochastic: bool,
    ) -> np.ndarray:
        with torch.no_grad():
            probabilities, _ = self.policy_outputs(state, valid_action_mask)
            action = (
                Categorical(probabilities).sample()
                if stochastic
                else probabilities.argmax(dim=-1)
            )
        return action.squeeze(0).cpu().numpy().astype(np.int64)

    def select_training_action(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_id: int,
        stochastic: bool = True,
    ) -> np.ndarray:
        """Select a training action; standard models ignore the regime label."""
        del regime_id
        return self.select_action(
            state, valid_action_mask, stochastic=stochastic
        )

    def select_validation_action(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_id: int,
        routing_mode: str = "learned",
        stochastic: bool = False,
    ) -> np.ndarray:
        """Select a validation action.

        Standard actors do not route by regime, so oracle and learned modes are
        identical.  Actor-MoE overrides this method.
        """
        del regime_id, routing_mode
        return self.select_action(state, valid_action_mask, stochastic=stochastic)

    def actor_parameters(self) -> List[nn.Parameter]:
        return list(self.actor.parameters())

    def critic_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError

    def value_parameters(self) -> List[nn.Parameter]:
        """Critic parameters updated by value/expert losses."""
        return self.critic_parameters()

    def gate_parameters(self) -> List[nn.Parameter]:
        """Critic-side gate parameters, separated for clipping/optimization."""
        return []

    def _init_standard_optimizers(self) -> None:
        self.actor_optimizer = optim.Adam(
            self.actor_parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.value_parameters(), lr=CRITIC_LR, eps=1e-5
        )
        self.gate_optimizer = None

    def _gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        continuation: torch.Tensor,
        elapsed_ticks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA), elapsed_ticks
        )
        trace_discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA * GAE_LAMBDA), elapsed_ticks
        )
        deltas = rewards + discounts * next_values * continuation - values
        advantages = torch.zeros_like(deltas)
        running = torch.zeros_like(deltas[-1])
        for index in reversed(range(deltas.size(0))):
            running = (
                deltas[index]
                + trace_discounts[index] * continuation[index] * running
            )
            advantages[index] = running
        returns = advantages + values

        # Advantage diagnostics use the exact SMDP-GAE values that feed the actor
        # before normalization. Rewards and critic values are in critic-scaled
        # units (CRITIC_REWARD_SCALE); the normalized/station advantages below are
        # the actual quantities used in the policy-gradient calculation.
        flat_advantage = advantages.detach().reshape(-1)
        if flat_advantage.numel() > 0:
            self._last_advantage_aux = {
                "mean_advantage": float(flat_advantage.mean().item()),
                "std_advantage": float(
                    flat_advantage.std(unbiased=False).item()
                ),
                "mean_abs_advantage": float(
                    flat_advantage.abs().mean().item()
                ),
                "min_advantage": float(flat_advantage.min().item()),
                "max_advantage": float(flat_advantage.max().item()),
                "positive_advantage_fraction": float(
                    (flat_advantage > 0.0).float().mean().item()
                ),
            }
        else:
            self._last_advantage_aux = {
                "mean_advantage": np.nan,
                "std_advantage": np.nan,
                "mean_abs_advantage": np.nan,
                "min_advantage": np.nan,
                "max_advantage": np.nan,
                "positive_advantage_fraction": np.nan,
            }
        return advantages, returns

    def _basic_td_target_and_error(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        continuation: torch.Tensor,
        elapsed_ticks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step SMDP TD target/error used by every A2C architecture.

        The event transition already stores the discounted reward accumulated over
        its primitive ticks.  Therefore the bootstrap term uses gamma**duration.
        No GAE or objective-specific return target is used in the basic-loss version.
        """
        discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA), elapsed_ticks
        )
        td_target = rewards + discounts * next_values.detach() * continuation
        td_error = td_target - values

        flat = td_error.detach().reshape(-1)
        if flat.numel() > 0:
            self._last_advantage_aux = {
                "mean_advantage": float(flat.mean().item()),
                "std_advantage": float(flat.std(unbiased=False).item()),
                "mean_abs_advantage": float(flat.abs().mean().item()),
                "min_advantage": float(flat.min().item()),
                "max_advantage": float(flat.max().item()),
                "positive_advantage_fraction": float(
                    (flat > 0.0).float().mean().item()
                ),
                # In this basic A2C version the actor uses the raw one-step TD
                # error, not a normalized GAE estimate.
                "mean_normalized_advantage": np.nan,
                "std_normalized_advantage": np.nan,
                "mean_abs_normalized_advantage": np.nan,
            }
        return td_target.detach(), td_error

    @staticmethod
    def _vector_gae_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        continuation: torch.Tensor,
        elapsed_ticks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SMDP-aware GAE for objective-value vectors without scalar diagnostics."""
        discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA), elapsed_ticks
        )
        trace_discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA * GAE_LAMBDA), elapsed_ticks
        )
        deltas = rewards + discounts * next_values * continuation - values
        advantages = torch.zeros_like(deltas)
        running = torch.zeros_like(deltas[-1])
        for index in reversed(range(deltas.size(0))):
            running = (
                deltas[index]
                + trace_discounts[index] * continuation[index] * running
            )
            advantages[index] = running
        return advantages, advantages + values

    @staticmethod
    def _aggregate_objective_values(
        objective_values: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate [B,K] objective values into one scalar value."""
        return (
            objective_values
            * OBJECTIVE_AGGREGATION_WEIGHTS_TENSOR.view(1, -1)
        ).sum(dim=1, keepdim=True)

    @staticmethod
    def _aggregate_expert_objective_values(
        expert_objective_values: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate [B,E,K] objective values into [B,E,1]."""
        return (
            expert_objective_values
            * OBJECTIVE_AGGREGATION_WEIGHTS_TENSOR.view(1, 1, -1)
        ).sum(dim=2, keepdim=True)

    @staticmethod
    def _semi_markov_mc_returns(
        rewards: torch.Tensor,
        continuation: torch.Tensor,
        elapsed_ticks: torch.Tensor,
    ) -> torch.Tensor:
        """Monte-Carlo returns with gamma**duration discounting.

        These targets are independent of the learned critic and therefore allow
        a fair learned-routing versus oracle-routing value comparison on exactly
        the same validation transitions.
        """
        discounts = torch.pow(
            torch.full_like(elapsed_ticks, GAMMA), elapsed_ticks
        )
        returns = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[-1])
        for index in reversed(range(rewards.size(0))):
            running = (
                rewards[index]
                + discounts[index] * continuation[index] * running
            )
            returns[index] = running
        return returns

    def oracle_routing_diagnostics(self, batch) -> Dict[str, float]:
        """Return empty diagnostics for models without a critic-side gate."""
        del batch
        return {}

    def _normalize_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        flat = advantage.detach().squeeze(-1)
        if flat.numel() > 1:
            flat = (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
        stats = dict(getattr(self, "_last_advantage_aux", {}))
        if flat.numel() > 0:
            stats.update(
                {
                    "mean_normalized_advantage": float(flat.mean().item()),
                    "std_normalized_advantage": float(
                        flat.std(unbiased=False).item()
                    ),
                    "mean_abs_normalized_advantage": float(
                        flat.abs().mean().item()
                    ),
                }
            )
        self._last_advantage_aux = stats
        return flat

    def _station_advantages(
        self,
        normalized_global_advantage: torch.Tensor,
        station_difference_reward: torch.Tensor,
        active_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Blend global GAE with a parameter-free station difference reward.

        The local term compares the dimensional setpoint error before and after a
        counterfactual one-station action. It reduces the variance caused by giving
        all simultaneously active stations exactly the same scalar credit.
        """
        active = active_masks.float()
        local = station_difference_reward.detach()
        selected = local[active > 0]
        if selected.numel() > 1:
            local_mean = selected.mean()
            local_std = selected.std(unbiased=False).clamp_min(1e-6)
            local = (local - local_mean) / local_std
        local = local.clamp(-3.0, 3.0)
        station_advantage = (
            normalized_global_advantage.unsqueeze(1)
            + STATION_DIFFERENCE_CREDIT_COEF * local
        )
        active_values = station_advantage.detach()[active > 0]
        stats = dict(getattr(self, "_last_advantage_aux", {}))
        if active_values.numel() > 0:
            stats.update(
                {
                    "mean_station_advantage": float(active_values.mean().item()),
                    "std_station_advantage": float(
                        active_values.std(unbiased=False).item()
                    ),
                    "mean_abs_station_advantage": float(
                        active_values.abs().mean().item()
                    ),
                }
            )
        self._last_advantage_aux = stats
        return station_advantage

    def _setpoint_guidance_loss(
        self,
        states: torch.Tensor,
        probabilities: torch.Tensor,
        valid_masks: torch.Tensor,
        active_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Low-weight, decaying physics guidance against maintain collapse.

        The first four state entries encode tanh(log(G_i/G_i_target)). They are
        inverted to recover the commanded setpoint ratio. Guidance requests an
        increase below the dimensional tolerance, a decrease above it, and
        maintain inside it. The coefficient decays to zero halfway through
        training, so the final policy remains reward-driven.
        """
        if SETPOINT_GUIDANCE_COEF <= 0.0:
            return probabilities.sum() * 0.0

        g_encoded = states[:, :NUM_STATIONS].clamp(-0.999, 0.999)
        ratios = torch.exp(torch.atanh(g_encoded))
        current_total = (ratios * STOCK_TARGET_TENSOR.view(1, -1)).sum(dim=1)

        targets = torch.full(
            (states.size(0), NUM_STATIONS),
            1,
            dtype=torch.long,
            device=states.device,
        )
        under = current_total < (TARGET_TOTAL_STOCK - DIMENSIONAL_TOLERANCE)
        over = current_total > (TARGET_TOTAL_STOCK + DIMENSIONAL_TOLERANCE)
        targets[under] = 2  # increase
        targets[over] = 0   # decrease

        target_valid = valid_masks.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)
        targets = torch.where(
            target_valid,
            targets,
            torch.ones_like(targets),
        )
        target_prob = probabilities.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1).clamp_min(1e-8)
        active = active_masks.float()
        return -(
            torch.log(target_prob) * active
        ).sum() / active.sum().clamp_min(1.0)

    def _guidance_scale(self) -> float:
        if self.training_progress >= SETPOINT_GUIDANCE_END_PROGRESS:
            return 0.0
        remaining = 1.0 - (
            self.training_progress / max(SETPOINT_GUIDANCE_END_PROGRESS, 1e-8)
        )
        return float(SETPOINT_GUIDANCE_COEF * max(0.0, remaining))

    def _standard_actor_terms(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_masks: torch.Tensor,
        active_masks: torch.Tensor,
        station_difference_reward: torch.Tensor,
        normalized_advantage: torch.Tensor,
        entropy_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, _ = self.policy_outputs(states, valid_masks)
        distribution = Categorical(probabilities)
        log_probability_station = distribution.log_prob(actions)
        active = active_masks.float()
        active_count = active.sum(dim=1).clamp_min(1.0)

        station_advantage = self._station_advantages(
            normalized_advantage,
            station_difference_reward,
            active,
        )
        policy_loss_per_transition = -(
            log_probability_station * station_advantage.detach() * active
        ).sum(dim=1) / active_count
        policy_loss = policy_loss_per_transition.mean()

        entropy = (distribution.entropy() * active).sum(dim=1) / active_count
        entropy_loss = -entropy_coefficient * entropy.mean()
        guidance_loss = self._setpoint_guidance_loss(
            states, probabilities, valid_masks, active_masks
        )
        mean_abs_credit = (
            station_difference_reward.abs() * active
        ).sum() / active.sum().clamp_min(1.0)
        return (
            policy_loss,
            entropy_loss,
            entropy.mean(),
            guidance_loss,
            mean_abs_credit,
        )

    def _finish_training_update(
        self,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
    ) -> Tuple[float, float, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if getattr(self, "gate_optimizer", None) is not None:
            self.gate_optimizer.zero_grad(set_to_none=True)

        critic_objective.backward()
        value_params = self.value_parameters()
        gate_params = self.gate_parameters()
        critic_g = grad_norm(value_params)
        gate_g = grad_norm(gate_params) if gate_params else np.nan
        torch.nn.utils.clip_grad_norm_(value_params, CRITIC_GRAD_CLIP)
        if gate_params:
            torch.nn.utils.clip_grad_norm_(gate_params, GATE_GRAD_CLIP)
        self.critic_optimizer.step()
        if getattr(self, "gate_optimizer", None) is not None:
            self.gate_optimizer.step()

        actor_objective.backward()
        actor_g = grad_norm(self.actor_parameters())
        torch.nn.utils.clip_grad_norm_(
            self.actor_parameters(), ACTOR_GRAD_CLIP
        )
        self.actor_optimizer.step()
        return actor_g, critic_g, gate_g

    def _validation_grad_norms(
        self,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
    ) -> Tuple[float, float, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if getattr(self, "gate_optimizer", None) is not None:
            self.gate_optimizer.zero_grad(set_to_none=True)

        critic_objective.backward()
        value_params = self.value_parameters()
        gate_params = self.gate_parameters()
        critic_g = grad_norm(value_params)
        gate_g = grad_norm(gate_params) if gate_params else np.nan

        self.critic_optimizer.zero_grad(set_to_none=True)
        if getattr(self, "gate_optimizer", None) is not None:
            self.gate_optimizer.zero_grad(set_to_none=True)

        actor_objective.backward()
        actor_g = grad_norm(self.actor_parameters())
        self.actor_optimizer.zero_grad(set_to_none=True)
        return actor_g, critic_g, gate_g

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def train_net(self, entropy_coefficient: float):
        values = self._loss_from_batch(
            self.make_batch(clear=True), entropy_coefficient
        )
        actor_loss, critic_loss, actor_obj, critic_obj, aux = values
        actor_g, critic_g, gate_g = self._finish_training_update(
            actor_obj, critic_obj
        )
        aux.update(getattr(self, "_last_advantage_aux", {}))
        aux.update(
            {
                "mean_actor_grad_norm": actor_g,
                "mean_critic_grad_norm": critic_g,
                "mean_gate_grad_norm": gate_g,
            }
        )
        return float(actor_loss.item()), float(critic_loss.item()), aux

    def eval_net_on_buffer(self, entropy_coefficient: float):
        values = self._loss_from_batch(
            self.make_batch(clear=False), entropy_coefficient
        )
        actor_loss, critic_loss, actor_obj, critic_obj, aux = values
        actor_g, critic_g, gate_g = self._validation_grad_norms(
            actor_obj, critic_obj
        )
        aux.update(getattr(self, "_last_advantage_aux", {}))
        aux.update(
            {
                "mean_actor_grad_norm": actor_g,
                "mean_critic_grad_norm": critic_g,
                "mean_gate_grad_norm": gate_g,
            }
        )
        return float(actor_loss.item()), float(critic_loss.item()), aux

    def gate_probabilities_np(self, states_np: np.ndarray) -> Optional[np.ndarray]:
        return None


# ---------------------------------------------------------------------
# Model 1: FC
# ---------------------------------------------------------------------
class A2C_FC(BaseA2C):
    def __init__(self, state_dim: int, **_: object) -> None:
        super().__init__()
        self.data = []
        self.training_progress = 0.0
        self.actor = FactorizedActorNetwork(state_dim)
        self.critic = mlp(state_dim, (74, 60), 1, layer_norm=True)
        self._init_standard_optimizers()

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.critic.parameters())

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states)

    def _loss_from_batch(self, batch, entropy_coefficient: float):
        (
            states, actions, rewards, reward_vectors, next_states, continuation,
            valid_masks, active_masks, durations, station_credits, regime_ids,
            next_regime_ids,
        ) = batch
        del reward_vectors, regime_ids, next_regime_ids

        values = self.value(states)
        with torch.no_grad():
            next_values = self.value(next_states)
        td_target, td_error = self._basic_td_target_and_error(
            rewards, values, next_values, continuation, durations
        )

        # Common critic objective for all five models: one-step SMDP TD MSE.
        critic_loss = F.mse_loss(values, td_target)
        (
            actor_loss, entropy_loss, policy_entropy, guidance_loss,
            mean_station_credit,
        ) = self._standard_actor_terms(
            states, actions, valid_masks, active_masks, station_credits,
            td_error.detach().squeeze(-1), entropy_coefficient,
        )
        aux = default_aux()
        aux["policy_entropy"] = float(policy_entropy.detach().item())
        aux["setpoint_guidance_loss"] = float(guidance_loss.detach().item())
        aux["mean_abs_station_credit"] = float(mean_station_credit.detach().item())
        return (
            actor_loss,
            critic_loss,
            actor_loss + entropy_loss + self._guidance_scale() * guidance_loss,
            critic_loss,
            aux,
        )

class A2C_RegimeHeads(BaseA2C):
    """Objective-specific MH critic with five one-to-one value heads.

    The legacy class name and model key are retained for result-file compatibility.
    No regime ID is used. Each head predicts one reward-component return, and the
    scalar value is their fixed linear aggregation.
    """

    def __init__(self, state_dim: int, **_: object) -> None:
        super().__init__()
        self.data = []
        self.training_progress = 0.0
        self.actor = FactorizedActorNetwork(state_dim)
        self.critic_trunk = nn.Sequential(
            nn.Linear(state_dim, 91),
            nn.LayerNorm(91),
            nn.ReLU(),
            nn.Linear(91, 43),
            nn.ReLU(),
        )
        self.objective_heads = nn.ModuleList(
            [nn.Linear(43, 1) for _ in range(NUM_OBJECTIVES)]
        )
        self._init_standard_optimizers()

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.critic_trunk.parameters()) + list(
            self.objective_heads.parameters()
        )

    def value_outputs(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.critic_trunk(states)
        objective_values = torch.cat(
            [head(latent) for head in self.objective_heads], dim=1
        )
        aggregated_value = self._aggregate_objective_values(objective_values)
        return aggregated_value, objective_values

    def _loss_from_batch(self, batch, entropy_coefficient: float):
        (
            states, actions, rewards, reward_vectors, next_states, continuation,
            valid_masks, active_masks, durations, station_credits, regime_ids,
            next_regime_ids,
        ) = batch
        del reward_vectors, regime_ids, next_regime_ids

        values, objective_values = self.value_outputs(states)
        with torch.no_grad():
            next_values, _ = self.value_outputs(next_states)
        td_target, td_error = self._basic_td_target_and_error(
            rewards, values, next_values, continuation, durations
        )

        # Same scalar TD-MSE objective as FC.  The five heads are architectural
        # components only; no objective-specific auxiliary loss is optimized.
        critic_loss = F.mse_loss(values, td_target)
        (
            actor_loss, entropy_loss, policy_entropy, guidance_loss,
            mean_station_credit,
        ) = self._standard_actor_terms(
            states, actions, valid_masks, active_masks, station_credits,
            td_error.detach().squeeze(-1), entropy_coefficient,
        )
        aux = default_aux()
        aux.update({
            "policy_entropy": float(policy_entropy.detach().item()),
            "setpoint_guidance_loss": float(guidance_loss.detach().item()),
            "mean_abs_station_credit": float(mean_station_credit.detach().item()),
        })
        for index, name in enumerate(OBJECTIVE_NAMES):
            aux[f"mean_value_{name}"] = float(
                objective_values[:, index].detach().mean().item()
            )
        return (
            actor_loss,
            critic_loss,
            actor_loss + entropy_loss + self._guidance_scale() * guidance_loss,
            critic_loss,
            aux,
        )

class CriticGateMixin:
    gate_scope = "critic"

    def _set_gate_temperature(self, progress: float) -> None:
        p = float(np.clip(progress, 0.0, 1.0))
        scheduled = self.gate_temp_start + p * (
            self.gate_temp_end - self.gate_temp_start
        )
        # Do not sharpen an inaccurate gate.  Before alignment is adequate, a
        # temperature floor of 1.0 avoids becoming confidently wrong.
        if self.gate_alignment_ema < GATE_ALIGNMENT_SHARPEN_THRESHOLD:
            scheduled = max(1.0, scheduled)
        self.gate_temperature = float(scheduled)

    def set_training_progress(self, progress: float) -> None:
        super().set_training_progress(progress)
        self._set_gate_temperature(progress)

    def _oracle_routing_weight(self) -> float:
        p = self.training_progress
        if p <= ORACLE_ROUTING_END_PROGRESS:
            return 1.0
        if p >= BLENDED_ROUTING_END_PROGRESS:
            return 0.0
        span = max(
            BLENDED_ROUTING_END_PROGRESS - ORACLE_ROUTING_END_PROGRESS,
            1e-8,
        )
        return float(
            (BLENDED_ROUTING_END_PROGRESS - p) / span
        )

    def _routing_probabilities(
        self,
        learned_probabilities: torch.Tensor,
        regime_ids: Optional[torch.Tensor],
        routing_mode: str = "curriculum",
    ) -> Tuple[torch.Tensor, float]:
        """Return routing weights for training or controlled validation.

        ``curriculum`` preserves the oracle-to-learned training schedule.
        ``learned`` forces the learned gate, while ``oracle`` selects the expert
        matching the true latent regime.
        """
        mode = str(routing_mode).lower()
        if mode not in {"curriculum", "learned", "oracle"}:
            raise ValueError(f"Unknown routing_mode={routing_mode}")
        if mode == "learned" or regime_ids is None:
            return learned_probabilities.detach(), 0.0
        oracle = F.one_hot(
            regime_ids, num_classes=NUM_REGIMES
        ).to(dtype=learned_probabilities.dtype)
        if mode == "oracle":
            return oracle, 1.0

        oracle_weight = self._oracle_routing_weight()
        if oracle_weight <= 0.0:
            return learned_probabilities.detach(), 0.0
        routing = (
            oracle_weight * oracle
            + (1.0 - oracle_weight) * learned_probabilities.detach()
        )
        return routing, float(oracle_weight)

    def oracle_routing_diagnostics(self, batch) -> Dict[str, float]:
        """Compare learned and oracle critic routing on fixed transitions."""
        (
            states,
            actions,
            rewards,
            reward_vectors,
            next_states,
            continuation,
            valid_masks,
            active_masks,
            durations,
            station_credits,
            regime_ids,
            next_regime_ids,
        ) = batch
        del (
            actions, reward_vectors, next_states, valid_masks, active_masks,
            station_credits, next_regime_ids
        )
        with torch.no_grad():
            learned_values = self.value_outputs(
                states, regime_ids, routing_mode="learned"
            )[0]
            oracle_values = self.value_outputs(
                states, regime_ids, routing_mode="oracle"
            )[0]
            mc_returns = self._semi_markov_mc_returns(
                rewards, continuation, durations
            )
            learned_huber = F.smooth_l1_loss(learned_values, mc_returns)
            oracle_huber = F.smooth_l1_loss(oracle_values, mc_returns)
            learned_mae = F.l1_loss(learned_values, mc_returns)
            oracle_mae = F.l1_loss(oracle_values, mc_returns)
            learned_mse = F.mse_loss(learned_values, mc_returns)
            oracle_mse = F.mse_loss(oracle_values, mc_returns)
            value_gap = (learned_values - oracle_values).abs().mean()
            improvement = 100.0 * (
                learned_huber - oracle_huber
            ) / learned_huber.clamp_min(1e-8)
        return {
            "critic_mc_huber_learned": float(learned_huber.item()),
            "critic_mc_huber_oracle": float(oracle_huber.item()),
            "critic_mc_mae_learned": float(learned_mae.item()),
            "critic_mc_mae_oracle": float(oracle_mae.item()),
            "critic_mc_mse_learned": float(learned_mse.item()),
            "critic_mc_mse_oracle": float(oracle_mse.item()),
            "oracle_critic_huber_improvement_pct": float(improvement.item()),
            "learned_oracle_value_abs_gap": float(value_gap.item()),
        }

    @staticmethod
    def _expert_correlation_loss(expert_values: torch.Tensor) -> torch.Tensor:
        """Penalize duplicate expert outputs without forcing arbitrary separation."""
        values = expert_values.squeeze(-1)  # [B, E]
        if values.size(0) < 4:
            return values.sum() * 0.0
        centered = values - values.mean(dim=0, keepdim=True)
        # Add epsilon before sqrt. This keeps the derivative finite when
        # zero-initialized residual experts produce identical zero outputs.
        scale = (centered.pow(2).mean(dim=0) + 1e-6).sqrt()
        normalized = centered / scale.unsqueeze(0)
        corr = normalized.transpose(0, 1) @ normalized / float(values.size(0))
        off_diagonal = corr - torch.diag_embed(torch.diagonal(corr))
        return off_diagonal.pow(2).sum() / max(
            1, NUM_REGIMES * (NUM_REGIMES - 1)
        )

    def _update_gate_alignment_ema(
        self, gate_terms: Dict[str, torch.Tensor]
    ) -> None:
        current = float(
            gate_terms["gate_alignment_tensor"].detach().item()
        )
        if not np.isfinite(self.gate_alignment_ema):
            self.gate_alignment_ema = current
        else:
            self.gate_alignment_ema = (
                0.95 * self.gate_alignment_ema + 0.05 * current
            )

    def _critic_gate_objective(
        self,
        value_loss: torch.Tensor,
        expert_loss: torch.Tensor,
        diversity_loss: torch.Tensor,
        gate_terms: Dict[str, torch.Tensor],
        value_loss_coef: float = 1.0,
        expert_loss_coef: Optional[float] = None,
        diversity_coef: Optional[float] = None,
        additional_value_loss: Optional[torch.Tensor] = None,
        additional_value_loss_coef: float = 0.0,
    ) -> torch.Tensor:
        """Combine value, specialization, diversity, and supervised gate losses.

        The optional coefficients support staged MH+MoE training and the residual
        Critic-MoE base-value objective without duplicating gate optimization code.
        """
        self._update_gate_alignment_ema(gate_terms)
        sharpening_enabled = (
            self.training_progress >= BLENDED_ROUTING_END_PROGRESS
            and self.gate_alignment_ema
            >= GATE_ALIGNMENT_SHARPEN_THRESHOLD
        )
        entropy_weight = (
            self.gate_entropy_coef if sharpening_enabled else 0.0
        )
        expert_coef = (
            self.expert_specialization_coef
            if expert_loss_coef is None
            else float(expert_loss_coef)
        )
        diversity_weight = (
            EXPERT_DIVERSITY_COEF
            if diversity_coef is None
            else float(diversity_coef)
        )
        objective = (
            float(value_loss_coef) * value_loss
            + expert_coef * expert_loss
            + diversity_weight * diversity_loss
            + self.gate_supervision_coef * gate_terms["gate_ce"]
            + self.gate_balance_coef * gate_terms["gate_balance"]
            + entropy_weight * gate_terms["gate_entropy_tensor"]
        )
        if additional_value_loss is not None and additional_value_loss_coef != 0.0:
            objective = objective + float(additional_value_loss_coef) * additional_value_loss
        return objective

    def _gate_aux(
        self,
        gate_terms: Dict[str, torch.Tensor],
        probabilities: torch.Tensor,
        expert_loss: torch.Tensor,
        diversity_loss: torch.Tensor,
        oracle_weight: float,
    ) -> Dict[str, float]:
        aux = default_aux()
        sharpening_enabled = float(
            self.training_progress >= BLENDED_ROUTING_END_PROGRESS
            and self.gate_alignment_ema
            >= GATE_ALIGNMENT_SHARPEN_THRESHOLD
        )
        aux.update(
            {
                "gate_ce": float(gate_terms["gate_ce"].detach().item()),
                "expert_td": float(expert_loss.detach().item()),
                "expert_correlation_loss": float(
                    diversity_loss.detach().item()
                ),
                "routing_oracle_weight": float(oracle_weight),
                "gate_entropy": float(
                    gate_terms["gate_entropy_tensor"].detach().item()
                ),
                "gate_balance": float(
                    gate_terms["gate_balance"].detach().item()
                ),
                "gate_acc": float(
                    gate_terms["gate_acc_tensor"].detach().item()
                ),
                "gate_regime_alignment": float(
                    gate_terms["gate_alignment_tensor"].detach().item()
                ),
                "gate_alignment_ema": float(self.gate_alignment_ema),
                "gate_sharpening_enabled": sharpening_enabled,
                "gate_temperature": float(self.gate_temperature),
            }
        )
        average = probabilities.detach().mean(dim=0).cpu().numpy()
        for index, probability in enumerate(average):
            aux[f"gate_p_{REGIME_NAMES[index]}"] = float(probability)
        return aux


# ---------------------------------------------------------------------
# Model 3: regime-value-head critic MoE
# ---------------------------------------------------------------------
class A2C_CriticMoE(CriticGateMixin, BaseA2C):
    """Critic MoE with one independent scalar value head per operating regime.

    Each regime expert first produces its own latent representation and then its
    own scalar value estimate.  The learned/oracle routing probabilities are
    applied to the four scalar regime values, not to the expert feature vectors.
    Thus the deployed critic is

        V(s) = sum_e rho_e(s) V_e(s),

    where e indexes normal, congestion, starvation, and quality-risk regimes.
    """

    def __init__(
        self,
        state_dim: int,
        gate_supervision_coef: float = 0.50,
        expert_specialization_coef: float = 0.50,
        gate_balance_coef: float = 0.01,
        gate_entropy_coef: float = 0.001,
        gate_temp_start: float = 1.5,
        gate_temp_end: float = 0.9,
        **_: object,
    ) -> None:
        super().__init__()
        self.data = []
        self.training_progress = 0.0
        self.gate_alignment_ema = np.nan
        self.actor = FactorizedActorNetwork(state_dim)

        # Shared state encoder: 19,605 parameters.
        self.value_encoder = nn.Sequential(
            nn.Linear(state_dim, 36),
            nn.LayerNorm(36),
            nn.ReLU(),
            nn.Linear(36, 42),
            nn.ReLU(),
        )
        # Four regime-specific feature experts: 38,008 parameters total.
        self.regime_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(42, 11),
                    nn.ReLU(),
                    nn.Linear(11, 6),
                )
                for _ in range(NUM_REGIMES)
            ]
        )
        # Four independent regime-specific scalar value heads: 92 parameters.
        self.regime_value_heads = nn.ModuleList(
            [nn.Linear(6, 1) for _ in range(NUM_REGIMES)]
        )

        # State-dependent regime gate: 11,391 parameters.
        self.gate_encoder = nn.Sequential(
            nn.Linear(state_dim, 41),
            nn.LayerNorm(41),
            nn.ReLU(),
            nn.Linear(41, 18),
            nn.ReLU(),
        )
        self.regime_gate = nn.Linear(18, NUM_REGIMES)

        self.gate_supervision_coef = float(gate_supervision_coef)
        self.expert_specialization_coef = float(expert_specialization_coef)
        self.gate_balance_coef = float(gate_balance_coef)
        self.gate_entropy_coef = float(gate_entropy_coef)
        self.gate_temp_start = float(gate_temp_start)
        self.gate_temp_end = float(gate_temp_end)
        self.gate_temperature = self.gate_temp_start

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.value_parameters(), lr=CRITIC_LR, eps=1e-5
        )
        self.gate_optimizer = optim.Adam(
            self.gate_parameters(), lr=GATE_LR, eps=1e-5
        )

    def critic_parameters(self) -> List[nn.Parameter]:
        return self.value_parameters() + self.gate_parameters()

    def value_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.value_encoder.parameters())
            + list(self.regime_experts.parameters())
            + list(self.regime_value_heads.parameters())
        )

    def gate_parameters(self) -> List[nn.Parameter]:
        return list(self.gate_encoder.parameters()) + list(
            self.regime_gate.parameters()
        )

    def value_outputs(
        self,
        states: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        routing_mode: str = "curriculum",
    ):
        shared_latent = self.value_encoder(states)
        expert_features = torch.stack(
            [expert(shared_latent) for expert in self.regime_experts], dim=1
        )  # [B, E, D]

        # Each expert has its own scalar value head before any gating occurs.
        expert_values = torch.stack(
            [
                head(expert_features[:, index, :])
                for index, head in enumerate(self.regime_value_heads)
            ],
            dim=1,
        )  # [B, E, 1]

        gate_latent = self.gate_encoder(states)
        gate_logits = self.regime_gate(gate_latent)
        learned_probabilities = F.softmax(
            gate_logits / max(self.gate_temperature, 1e-6), dim=-1
        )
        routing_probabilities, oracle_weight = self._routing_probabilities(
            learned_probabilities, regime_ids, routing_mode=routing_mode
        )

        # Gate the regime-specific scalar values directly.
        mixed_value = (
            expert_values * routing_probabilities.unsqueeze(-1)
        ).sum(dim=1)
        return (
            mixed_value,
            learned_probabilities,
            expert_values,
            gate_logits,
            oracle_weight,
            expert_features,
        )

    def _loss_from_batch(self, batch, entropy_coefficient: float):
        (
            states, actions, rewards, reward_vectors, next_states, continuation,
            valid_masks, active_masks, durations, station_credits, regime_ids,
            next_regime_ids,
        ) = batch
        del reward_vectors
        batch_index = torch.arange(states.size(0), device=states.device)

        values, gate_probs, expert_values, gate_logits, oracle_weight, _ = (
            self.value_outputs(states, regime_ids)
        )
        with torch.no_grad():
            next_values, _, _, _, _, _ = self.value_outputs(
                next_states, next_regime_ids
            )
        td_target, td_error = self._basic_td_target_and_error(
            rewards, values, next_values, continuation, durations
        )

        # Common scalar critic loss shared with FC/MH/MH+MoE/Actor-MoE.
        critic_loss = F.mse_loss(values, td_target)

        # Gate CE is optimized separately from the A2C actor/critic losses so that
        # the regime labels retain their interpretation.  Expert/diversity terms
        # below are diagnostics only and do not enter the optimization objective.
        gate_terms = gate_losses_and_metrics(
            gate_logits, gate_probs, regime_ids, NUM_REGIMES
        )
        self._update_gate_alignment_ema(gate_terms)
        gate_loss = self.gate_supervision_coef * gate_terms["gate_ce"]
        critic_objective = critic_loss + gate_loss

        assigned_values = expert_values[batch_index, regime_ids, :]
        expert_td_diagnostic = F.mse_loss(
            assigned_values.detach(), td_target.detach()
        )
        diversity_diagnostic = self._expert_correlation_loss(
            expert_values.detach()
        )

        (
            actor_loss, entropy_loss, policy_entropy, guidance_loss,
            mean_station_credit,
        ) = self._standard_actor_terms(
            states, actions, valid_masks, active_masks, station_credits,
            td_error.detach().squeeze(-1), entropy_coefficient,
        )
        aux = self._gate_aux(
            gate_terms, gate_probs, expert_td_diagnostic,
            diversity_diagnostic, oracle_weight
        )
        aux["mixture_value_loss_coef"] = 1.0
        aux["assigned_expert_loss_coef"] = 0.0
        aux["head_diversity_coef"] = 0.0
        aux["policy_entropy"] = float(policy_entropy.detach().item())
        aux["setpoint_guidance_loss"] = float(guidance_loss.detach().item())
        aux["mean_abs_station_credit"] = float(mean_station_credit.detach().item())
        return (
            actor_loss,
            critic_loss,
            actor_loss + entropy_loss + self._guidance_scale() * guidance_loss,
            critic_objective,
            aux,
        )

    def gate_probabilities_np(
        self, states_np: np.ndarray
    ) -> Optional[np.ndarray]:
        with torch.no_grad():
            states = torch.tensor(
                states_np, dtype=torch.float32, device=device
            )
            _, probabilities, _, _, _, _ = self.value_outputs(
                states, regime_ids=None
            )
            return probabilities.cpu().numpy()


# ---------------------------------------------------------------------
# Model 4: MH + regime-value-head MoE
# ---------------------------------------------------------------------
class A2C_MultiHeadCriticMoE(CriticGateMixin, BaseA2C):
    """MH critic followed by regime-specific value heads and learned gating.

    The first stage preserves the MH mechanism: a shared critic trunk feeds five
    objective-specific scalar heads for yield, Type-I defects, Type-II defects,
    buffer change, and lost orders.  Their five scalar predictions are then fused
    by a fully connected layer.  Four regime-specific scalar value heads consume
    that fused objective representation, and the final critic value is the gate-
    weighted sum of those four regime values.

    The hidden widths are reduced relative to the standalone MH critic only to
    reserve parameters for the fusion layer, regime heads, and gate while keeping
    the same approximately 10k total trainable-parameter budget.
    """

    def __init__(
        self,
        state_dim: int,
        gate_supervision_coef: float = 0.50,
        expert_specialization_coef: float = 0.50,
        gate_balance_coef: float = 0.01,
        gate_entropy_coef: float = 0.001,
        gate_temp_start: float = 1.5,
        gate_temp_end: float = 0.9,
        **_: object,
    ) -> None:
        super().__init__()
        self.data = []
        self.training_progress = 0.0
        self.gate_alignment_ema = np.nan
        self.actor = FactorizedActorNetwork(state_dim)

        # MH-style common trunk.  Topology matches the standalone MH critic;
        # widths are budget-adjusted so the full MH+MoE model remains ~100k.
        # Trunk: 55,830 parameters.
        self.objective_trunk = nn.Sequential(
            nn.Linear(state_dim, 46),
            nn.LayerNorm(46),
            nn.ReLU(),
            nn.Linear(46, 62),
            nn.ReLU(),
        )
        # Exactly the same five-head mechanism as MH: one scalar head per reward
        # component.  Five heads: 1,080 parameters.
        self.objective_heads = nn.ModuleList(
            [nn.Linear(62, 1) for _ in range(NUM_OBJECTIVES)]
        )

        # Objective values are fused before any regime specialization.
        # 5 -> 64 FC + LayerNorm: 512 parameters.
        self.objective_fusion = nn.Sequential(
            nn.Linear(NUM_OBJECTIVES, 19),
            nn.LayerNorm(19),
            nn.ReLU(),
        )
        # Four regime-specific scalar value heads: 260 parameters.
        self.regime_value_heads = nn.ModuleList(
            [nn.Linear(19, 1) for _ in range(NUM_REGIMES)]
        )

        # Independent state-dependent regime gate: 11,391 parameters.
        self.gate_encoder = nn.Sequential(
            nn.Linear(state_dim, 42),
            nn.LayerNorm(42),
            nn.ReLU(),
            nn.Linear(42, 17),
            nn.ReLU(),
        )
        self.regime_gate = nn.Linear(17, NUM_REGIMES)

        self.gate_supervision_coef = float(gate_supervision_coef)
        self.expert_specialization_coef = float(expert_specialization_coef)
        self.gate_balance_coef = float(gate_balance_coef)
        self.gate_entropy_coef = float(gate_entropy_coef)
        self.gate_temp_start = float(gate_temp_start)
        self.gate_temp_end = float(gate_temp_end)
        self.gate_temperature = self.gate_temp_start

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.value_parameters(), lr=CRITIC_LR, eps=1e-5
        )
        self.gate_optimizer = optim.Adam(
            self.gate_parameters(), lr=GATE_LR, eps=1e-5
        )

    def critic_parameters(self) -> List[nn.Parameter]:
        return self.value_parameters() + self.gate_parameters()

    def value_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.objective_trunk.parameters())
            + list(self.objective_heads.parameters())
            + list(self.objective_fusion.parameters())
            + list(self.regime_value_heads.parameters())
        )

    def gate_parameters(self) -> List[nn.Parameter]:
        return list(self.gate_encoder.parameters()) + list(
            self.regime_gate.parameters()
        )

    def _objective_values(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(latent) for head in self.objective_heads], dim=1)

    def value_outputs(
        self,
        states: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        routing_mode: str = "curriculum",
    ):
        # Stage 1: MH objective decomposition occurs first.
        objective_latent = self.objective_trunk(states)
        objective_values = self._objective_values(objective_latent)  # [B, K]

        # Stage 2: combine the five objective values with a fully connected layer.
        fused_objective = self.objective_fusion(objective_values)  # [B, F]

        # Stage 3: independent regime-specific scalar value heads.
        regime_values = torch.stack(
            [head(fused_objective) for head in self.regime_value_heads], dim=1
        )  # [B, E, 1]

        # Stage 4: learned/oracle gate produces the final weighted value sum.
        gate_latent = self.gate_encoder(states)
        gate_logits = self.regime_gate(gate_latent)
        learned_probabilities = F.softmax(
            gate_logits / max(self.gate_temperature, 1e-6), dim=-1
        )
        routing_probabilities, oracle_weight = self._routing_probabilities(
            learned_probabilities, regime_ids, routing_mode=routing_mode
        )
        mixed_scalar_value = (
            regime_values * routing_probabilities.unsqueeze(-1)
        ).sum(dim=1)

        return (
            mixed_scalar_value,
            learned_probabilities,
            regime_values,
            gate_logits,
            oracle_weight,
            objective_values,
        )

    def _loss_from_batch(self, batch, entropy_coefficient: float):
        (
            states, actions, rewards, reward_vectors, next_states, continuation,
            valid_masks, active_masks, durations, station_credits, regime_ids,
            next_regime_ids,
        ) = batch
        del reward_vectors
        batch_index = torch.arange(states.size(0), device=states.device)

        values, gate_probs, regime_values, gate_logits, oracle_weight, objective_values = (
            self.value_outputs(states, regime_ids)
        )
        with torch.no_grad():
            next_values, _, _, _, _, _ = self.value_outputs(
                next_states, next_regime_ids
            )
        td_target, td_error = self._basic_td_target_and_error(
            rewards, values, next_values, continuation, durations
        )

        # Same basic scalar TD-MSE critic loss used by all five architectures.
        # The MH objective heads and MoE regime heads are trained only through the
        # final scalar value path; no vector-return, consistency, expert-TD, or
        # diversity auxiliary loss is added.
        critic_loss = F.mse_loss(values, td_target)

        gate_terms = gate_losses_and_metrics(
            gate_logits, gate_probs, regime_ids, NUM_REGIMES
        )
        self._update_gate_alignment_ema(gate_terms)
        gate_loss = self.gate_supervision_coef * gate_terms["gate_ce"]
        critic_objective = critic_loss + gate_loss

        assigned_values = regime_values[batch_index, regime_ids, :]
        expert_td_diagnostic = F.mse_loss(
            assigned_values.detach(), td_target.detach()
        )
        diversity_diagnostic = self._expert_correlation_loss(
            regime_values.detach()
        )

        (
            actor_loss, entropy_loss, policy_entropy, guidance_loss,
            mean_station_credit,
        ) = self._standard_actor_terms(
            states, actions, valid_masks, active_masks, station_credits,
            td_error.detach().squeeze(-1), entropy_coefficient,
        )
        aux = self._gate_aux(
            gate_terms, gate_probs, expert_td_diagnostic,
            diversity_diagnostic, oracle_weight
        )
        aux.update({
            "mixture_value_loss_coef": 1.0,
            "assigned_expert_loss_coef": 0.0,
            "head_diversity_coef": 0.0,
            "policy_entropy": float(policy_entropy.detach().item()),
            "setpoint_guidance_loss": float(guidance_loss.detach().item()),
            "mean_abs_station_credit": float(mean_station_credit.detach().item()),
        })
        for index, name in enumerate(OBJECTIVE_NAMES):
            aux[f"mean_value_{name}"] = float(
                objective_values[:, index].detach().mean().item()
            )
        return (
            actor_loss,
            critic_loss,
            actor_loss + entropy_loss + self._guidance_scale() * guidance_loss,
            critic_objective,
            aux,
        )

    def gate_probabilities_np(
        self, states_np: np.ndarray
    ) -> Optional[np.ndarray]:
        with torch.no_grad():
            states = torch.tensor(
                states_np, dtype=torch.float32, device=device
            )
            outputs = self.value_outputs(states, regime_ids=None)
            return outputs[1].cpu().numpy()


# ---------------------------------------------------------------------
# Model 5: Actor MoE + FC critic
# ---------------------------------------------------------------------
class A2C_ActorMoE(BaseA2C):
    """Residual Actor-MoE with supervised routing and on-policy curriculum.

    Training behavior follows the same oracle/blended/learned distribution used
    in the policy loss.  This removes the previous off-policy auxiliary expert
    loss, where actions sampled from the mixture were incorrectly treated as if
    they had been sampled by the true-regime expert.
    """

    gate_scope = "actor"

    def __init__(
        self,
        state_dim: int,
        gate_supervision_coef: float = 0.40,
        actor_expert_specialization_coef: float = 0.0,
        gate_balance_coef: float = 0.01,
        gate_entropy_coef: float = 0.001,
        gate_temp_start: float = 1.5,
        gate_temp_end: float = 0.9,
        actor_residual_scale: float = 1.5,
        actor_residual_l2_coef: float = 0.0,
        actor_residual_lr_multiplier: float = ACTOR_RESIDUAL_LR_MULTIPLIER,
        **_: object,
    ) -> None:
        super().__init__()
        self.data = []
        self.training_progress = 0.0
        self.gate_alignment_ema = np.nan
        self.actor = ActorMoENetwork(
            state_dim,
            num_experts=NUM_REGIMES,
            gate_temp_start=gate_temp_start,
            gate_temp_end=gate_temp_end,
            residual_scale=actor_residual_scale,
        )
        self.critic = mlp(state_dim, (74, 60), 1, layer_norm=True)
        self.gate_supervision_coef = float(gate_supervision_coef)
        self.gate_balance_coef = float(gate_balance_coef)
        self.gate_entropy_coef = float(gate_entropy_coef)
        self.actor_residual_l2_coef = float(actor_residual_l2_coef)
        self.actor_residual_lr_multiplier = float(actor_residual_lr_multiplier)
        # Kept only for backward-compatible command lines.  The biased auxiliary
        # expert policy loss has deliberately been removed in v7.
        self.deprecated_actor_expert_specialization_coef = float(
            actor_expert_specialization_coef
        )

        self.actor_optimizer = optim.Adam(
            [
                {
                    "params": list(self.actor.policy_encoder.parameters())
                    + list(self.actor.base_head.parameters()),
                    "lr": ACTOR_LR,
                },
                {
                    "params": self.actor.residual_experts.parameters(),
                    "lr": ACTOR_LR * self.actor_residual_lr_multiplier,
                },
            ],
            eps=1e-5,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=CRITIC_LR, eps=1e-5
        )
        self.gate_optimizer = optim.Adam(
            self.actor.gate_parameters(), lr=GATE_LR, eps=1e-5
        )

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.critic.parameters())

    def actor_policy_parameters(self) -> List[nn.Parameter]:
        return self.actor.policy_parameters()

    def actor_gate_parameters(self) -> List[nn.Parameter]:
        return self.actor.gate_parameters()

    def _oracle_routing_weight(self) -> float:
        progress = self.training_progress
        if progress <= ORACLE_ROUTING_END_PROGRESS:
            return 1.0
        if progress >= BLENDED_ROUTING_END_PROGRESS:
            return 0.0
        span = max(
            BLENDED_ROUTING_END_PROGRESS - ORACLE_ROUTING_END_PROGRESS,
            1e-8,
        )
        return float(
            (BLENDED_ROUTING_END_PROGRESS - progress) / span
        )

    def set_training_progress(self, progress: float) -> None:
        super().set_training_progress(progress)
        scheduled = (
            self.actor.gate_temp_start
            + self.training_progress
            * (self.actor.gate_temp_end - self.actor.gate_temp_start)
        )
        if (
            not np.isfinite(self.gate_alignment_ema)
            or self.gate_alignment_ema
            < GATE_ALIGNMENT_SHARPEN_THRESHOLD
        ):
            scheduled = max(1.0, scheduled)
        self.actor.gate_temperature = float(scheduled)

    def policy_outputs(
        self,
        states: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ):
        outputs = self.actor(
            states,
            valid_action_mask,
            regime_ids=None,
            oracle_weight=0.0,
            routing_mode="learned",
        )
        return outputs[0], outputs[1:]

    def select_training_action(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_id: int,
        stochastic: bool = True,
    ) -> np.ndarray:
        """Use the actual oracle-to-blended-to-learned behavior curriculum."""
        with torch.no_grad():
            regime_tensor = torch.full(
                (state.size(0),),
                int(regime_id),
                dtype=torch.long,
                device=state.device,
            )
            probabilities = self.actor(
                state,
                valid_action_mask,
                regime_ids=regime_tensor,
                oracle_weight=self._oracle_routing_weight(),
                routing_mode="curriculum",
            )[0]
            action = (
                Categorical(probabilities).sample()
                if stochastic
                else probabilities.argmax(dim=-1)
            )
        return action.squeeze(0).cpu().numpy().astype(np.int64)

    def select_validation_action(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        regime_id: int,
        routing_mode: str = "learned",
        stochastic: bool = False,
    ) -> np.ndarray:
        mode = str(routing_mode).lower()
        with torch.no_grad():
            regime_tensor = torch.full(
                (state.size(0),),
                int(regime_id),
                dtype=torch.long,
                device=state.device,
            )
            probabilities = self.actor(
                state,
                valid_action_mask,
                regime_ids=regime_tensor if mode == "oracle" else None,
                oracle_weight=1.0 if mode == "oracle" else 0.0,
                routing_mode=mode,
            )[0]
            action = (
                Categorical(probabilities).sample()
                if stochastic
                else probabilities.argmax(dim=-1)
            )
        return action.squeeze(0).cpu().numpy().astype(np.int64)

    def _loss_from_batch(self, batch, entropy_coefficient: float):
        (
            states, actions, rewards, reward_vectors, next_states, continuation,
            valid_masks, active_masks, durations, station_credits, regime_ids,
            next_regime_ids,
        ) = batch
        del reward_vectors, next_regime_ids

        values = self.critic(states)
        with torch.no_grad():
            next_values = self.critic(next_states)
        td_target, td_error = self._basic_td_target_and_error(
            rewards, values, next_values, continuation, durations
        )
        critic_loss = F.mse_loss(values, td_target)

        oracle_weight = self._oracle_routing_weight()
        (
            routed_probs, gate_probs, residual_logits, gate_logits,
            residual_alpha, routing_weights,
        ) = self.actor(
            states, valid_masks, regime_ids=regime_ids,
            oracle_weight=oracle_weight, routing_mode="curriculum",
        )

        # Same basic advantage-weighted log-likelihood used by every actor.
        distribution = Categorical(routed_probs)
        active = active_masks.float()
        active_count = active.sum(dim=1).clamp_min(1.0)
        log_probability = distribution.log_prob(actions)
        station_advantage = self._station_advantages(
            td_error.detach().squeeze(-1), station_credits, active_masks
        )
        actor_loss = -(
            (log_probability * station_advantage.detach() * active).sum(dim=1)
            / active_count
        ).mean()
        policy_entropy = (distribution.entropy() * active).sum(dim=1) / active_count
        entropy_loss = -entropy_coefficient * policy_entropy.mean()

        gate_terms = gate_losses_and_metrics(
            gate_logits, gate_probs, regime_ids, NUM_REGIMES
        )
        current_alignment = float(
            gate_terms["gate_alignment_tensor"].detach().item()
        )
        if not np.isfinite(self.gate_alignment_ema):
            self.gate_alignment_ema = current_alignment
        else:
            self.gate_alignment_ema = (
                0.95 * self.gate_alignment_ema + 0.05 * current_alignment
            )

        # Gate CE is the only structure-specific optimized auxiliary term and is
        # separated from both reported actor loss and reported critic loss.
        gate_loss = self.gate_supervision_coef * gate_terms["gate_ce"]
        critic_objective = critic_loss + gate_loss
        actor_objective = actor_loss + entropy_loss

        residual_l2 = residual_logits.pow(2).mean().detach()
        mean_station_credit = (
            station_credits.abs() * active
        ).sum() / active.sum().clamp_min(1.0)
        aux = default_aux()
        aux.update({
            "gate_ce": float(gate_terms["gate_ce"].detach().item()),
            "expert_td": 0.0,
            "routing_oracle_weight": float(oracle_weight),
            "gate_entropy": float(gate_terms["gate_entropy_tensor"].detach().item()),
            "gate_balance": float(gate_terms["gate_balance"].detach().item()),
            "gate_acc": float(gate_terms["gate_acc_tensor"].detach().item()),
            "gate_regime_alignment": current_alignment,
            "gate_alignment_ema": float(self.gate_alignment_ema),
            "gate_sharpening_enabled": 0.0,
            "gate_temperature": float(self.actor.gate_temperature),
            "policy_entropy": float(policy_entropy.mean().detach().item()),
            "setpoint_guidance_loss": 0.0,
            "mean_abs_station_credit": float(mean_station_credit.detach().item()),
            "actor_residual_l2": float(residual_l2.item()),
            "actor_residual_alpha": float(residual_alpha.mean().detach().item()),
            "actor_routing_confidence": float(
                routing_weights.max(dim=1).values.mean().detach().item()
            ),
        })
        average_gate = gate_probs.detach().mean(dim=0).cpu().numpy()
        for index, probability in enumerate(average_gate):
            aux[f"gate_p_{REGIME_NAMES[index]}"] = float(probability)

        return actor_loss, critic_loss, actor_objective, critic_objective, aux

    def _finish_training_update(
        self,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
    ) -> Tuple[float, float, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)

        # critic_objective = common TD-MSE + separate supervised gate CE.
        critic_objective.backward()
        critic_params = self.value_parameters()
        gate_params = self.actor_gate_parameters()
        critic_g = grad_norm(critic_params)
        gate_g = grad_norm(gate_params)
        torch.nn.utils.clip_grad_norm_(critic_params, CRITIC_GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(gate_params, GATE_GRAD_CLIP)
        self.critic_optimizer.step()
        self.gate_optimizer.step()

        # Actor gradient intentionally excludes the gate so its norm is directly
        # comparable with the other four models.
        actor_objective.backward()
        policy_params = self.actor_policy_parameters()
        actor_g = grad_norm(policy_params)
        torch.nn.utils.clip_grad_norm_(policy_params, ACTOR_GRAD_CLIP)
        self.actor_optimizer.step()
        return actor_g, critic_g, gate_g

    def _validation_grad_norms(
        self,
        actor_objective: torch.Tensor,
        critic_objective: torch.Tensor,
    ) -> Tuple[float, float, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)

        critic_objective.backward()
        critic_g = grad_norm(self.value_parameters())
        gate_g = grad_norm(self.actor_gate_parameters())
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)

        actor_objective.backward()
        actor_g = grad_norm(self.actor_policy_parameters())
        self.actor_optimizer.zero_grad(set_to_none=True)
        return actor_g, critic_g, gate_g

    def gate_probabilities_np(
        self, states_np: np.ndarray
    ) -> Optional[np.ndarray]:
        with torch.no_grad():
            states = torch.tensor(
                states_np, dtype=torch.float32, device=device
            )
            probabilities, _ = self.actor.gate_outputs(states)
            return probabilities.cpu().numpy()


# ---------------------------------------------------------------------
# Shared model utilities
# ---------------------------------------------------------------------
def default_aux() -> Dict[str, float]:
    return {
        "gate_ce": 0.0,
        "expert_td": 0.0,
        "expert_correlation_loss": np.nan,
        "routing_oracle_weight": np.nan,
        "gate_entropy": np.nan,
        "gate_balance": np.nan,
        "gate_acc": np.nan,
        "gate_regime_alignment": np.nan,
        "gate_alignment_ema": np.nan,
        "gate_sharpening_enabled": np.nan,
        "gate_temperature": np.nan,
        "policy_entropy": np.nan,
        "setpoint_guidance_loss": np.nan,
        "mean_abs_station_credit": np.nan,
        "mean_gate_grad_norm": np.nan,
        "actor_residual_l2": np.nan,
        "actor_residual_alpha": np.nan,
        "actor_routing_confidence": np.nan,
        "base_value_loss": np.nan,
        "residual_value_l2": np.nan,
        "mixture_value_loss_coef": np.nan,
        "assigned_expert_loss_coef": np.nan,
        "head_diversity_coef": np.nan,
        "objective_value_loss": np.nan,
        "scalar_consistency_loss": np.nan,
        "mean_value_step_yield": np.nan,
        "mean_value_step_type1": np.nan,
        "mean_value_step_type2": np.nan,
        "mean_value_step_buffer_change": np.nan,
        "mean_value_step_lost_orders": np.nan,
        "mean_advantage": np.nan,
        "std_advantage": np.nan,
        "mean_abs_advantage": np.nan,
        "min_advantage": np.nan,
        "max_advantage": np.nan,
        "positive_advantage_fraction": np.nan,
        "mean_normalized_advantage": np.nan,
        "std_normalized_advantage": np.nan,
        "mean_abs_normalized_advantage": np.nan,
        "mean_station_advantage": np.nan,
        "std_station_advantage": np.nan,
        "mean_abs_station_advantage": np.nan,
    }


MODEL_CLASSES: Dict[str, Type[BaseA2C]] = {
    "fc": A2C_FC,
    "regime_heads": A2C_RegimeHeads,
    "critic_moe": A2C_CriticMoE,
    "multihead_critic_moe": A2C_MultiHeadCriticMoE,
    "actor_moe": A2C_ActorMoE,
}


def entropy_coefficient(episode_index: int, total_episodes: int) -> float:
    progress = min(1.0, episode_index / max(1, int(0.70 * total_episodes)))
    return float(ENTROPY_START + progress * (ENTROPY_END - ENTROPY_START))


def build_scenario_splits(
    seed: int,
    num_scenarios: int,
    validation_ratio: float,
) -> Tuple[List[int], List[int]]:
    seeds = np.arange(seed, seed + num_scenarios, dtype=np.int64)
    rng = np.random.default_rng(seed + 777)
    rng.shuffle(seeds)
    validation_count = max(1, int(round(num_scenarios * validation_ratio)))
    return seeds[validation_count:].tolist(), seeds[:validation_count].tolist()


def instantiate_model(
    model_name: str,
    state_dim: int,
    args: argparse.Namespace,
) -> BaseA2C:
    model_class = MODEL_CLASSES[model_name]
    return model_class(
        state_dim=state_dim,
        gate_supervision_coef=args.gate_supervision_coef,
        expert_specialization_coef=args.expert_specialization_coef,
        actor_expert_specialization_coef=args.actor_expert_specialization_coef,
        gate_balance_coef=args.gate_balance_coef,
        gate_entropy_coef=args.gate_entropy_coef,
        gate_temp_start=args.gate_temp_start,
        gate_temp_end=args.gate_temp_end,
        actor_residual_scale=args.actor_residual_scale,
        actor_residual_l2_coef=args.actor_residual_l2_coef,
        actor_residual_lr_multiplier=args.actor_residual_lr_multiplier,
    ).to(device)


def make_transition(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    done: bool,
    truncated: bool,
    valid_action_mask: np.ndarray,
    active_station_mask: np.ndarray,
    regime_id: int,
    next_info: Dict,
):
    primitive_rewards = np.asarray(next_info["primitive_rewards"], dtype=np.float32)
    primitive_vectors = np.asarray(
        next_info["primitive_reward_vectors"], dtype=np.float32
    )
    # Scale critic targets while keeping raw environment returns for reporting.
    discounted_reward = (
        float(discounted_interval_values(primitive_rewards))
        * CRITIC_REWARD_SCALE
    )
    discounted_vector = (
        np.asarray(
            discounted_interval_values(primitive_vectors),
            dtype=np.float32,
        )
        * CRITIC_REWARD_SCALE
    )
    station_difference_reward = np.zeros(NUM_STATIONS, dtype=np.float32)
    return (
        state,
        action,
        discounted_reward,
        discounted_vector,
        next_state,
        truncated,
        done,
        valid_action_mask,
        active_station_mask,
        int(next_info["elapsed_ticks"]),
        station_difference_reward,
        int(regime_id),
        int(next_info["regime_id"]),
    )


def env_kwargs_from_args(args: argparse.Namespace) -> Dict:
    return {
        "num_stations": 4,
        "episode_horizon": args.horizon,
        "arrival_distribution": args.arrival_distribution,
        "base_arrival_rate": args.base_arrival_rate,
        "stock_ratio_min": args.stock_ratio_min,
        "stock_ratio_max": args.stock_ratio_max,
        "initial_stock_ratios": tuple(args.initial_stock_ratios),
        "process_time_cv": args.process_time_cv,
        "min_regime_duration": args.min_regime_duration,
        "max_regime_duration": args.max_regime_duration,
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def collect_actor_oracle_policy_validation(
    model: BaseA2C,
    env_kwargs: Dict,
    validation_seeds: List[int],
) -> pd.DataFrame:
    """Evaluate Actor-MoE while bypassing its learned gate.

    Critic-only MoE routing cannot change greedy deployment actions, so policy
    objective under oracle routing is meaningful only for Actor-MoE.
    """
    if model.gate_scope != "actor":
        return pd.DataFrame()
    rows: List[Dict] = []
    for scenario_index, scenario_seed in enumerate(validation_seeds, start=1):
        env = ManufacturingQueueRegimeEventEnv(**env_kwargs)
        state, info = env.reset(seed=int(scenario_seed))
        done = truncated = False
        total_reward = 0.0
        while not done and not truncated:
            regime_id = int(info["regime_id"])
            valid_mask = env.action_mask()
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            valid_tensor = torch.tensor(
                valid_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            action = model.select_validation_action(
                state_tensor,
                valid_tensor,
                regime_id=regime_id,
                routing_mode="oracle",
                stochastic=False,
            )
            state, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
        rows.append(
            {
                "scenario_index": int(scenario_index),
                "scenario_seed": int(scenario_seed),
                "oracle_objective_value": float(total_reward),
                "oracle_Y": float(info["Y"]),
                "oracle_type1_total": float(info["type1_total"]),
                "oracle_type2_total": float(info["type2_total"]),
                "oracle_buffer_total": float(info["buffer_total"]),
                "oracle_lost_orders": float(info["lost_orders"]),
                "oracle_decision_count": int(info["decision_count"]),
            }
        )
        env.close()
    return pd.DataFrame(rows)


def collect_validation_after_training(
    model: BaseA2C,
    env_kwargs: Dict,
    validation_seeds: List[int],
    replication_id: int,
    model_name: str,
    entropy_coeff: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    scenario_rows: List[Dict] = []
    gate_rows: List[Dict] = []
    action_rows: List[Dict] = []
    all_transitions: List[tuple] = []

    for scenario_index, scenario_seed in enumerate(validation_seeds, start=1):
        env = ManufacturingQueueRegimeEventEnv(**env_kwargs)
        state, info = env.reset(seed=int(scenario_seed))
        done = truncated = False
        total_reward = 0.0
        regime_counts = np.zeros(NUM_REGIMES, dtype=np.int64)
        elapsed_ticks: List[int] = []
        states_for_gate: List[np.ndarray] = []
        regimes_for_gate: List[int] = []
        decision_index = 0

        while not done and not truncated:
            regime_id = int(info["regime_id"])
            regime_counts[regime_id] += 1
            valid_mask = env.action_mask()
            active_mask = env.active_station_mask()
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            valid_tensor = torch.tensor(
                valid_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            action = model.select_action(
                state_tensor, valid_tensor, stochastic=False
            )
            next_state, reward, done, truncated, next_info = env.step(action)
            transition = make_transition(
                state,
                action,
                next_state,
                done,
                truncated,
                valid_mask,
                active_mask,
                regime_id,
                next_info,
            )
            all_transitions.append(transition)
            total_reward += float(reward)
            elapsed_ticks.append(int(next_info["elapsed_ticks"]))
            states_for_gate.append(state.astype(np.float32, copy=True))
            regimes_for_gate.append(regime_id)

            row = {
                "split": "validation",
                "replication": int(replication_id),
                "model": model_name,
                "model_label": model_label(model_name),
                "scenario_index": int(scenario_index),
                "scenario_seed": int(scenario_seed),
                "decision": int(decision_index),
                "regime_id": regime_id,
                "regime_name": REGIME_NAMES[regime_id],
                "next_regime_id": int(next_info["regime_id"]),
                "elapsed_ticks": int(next_info["elapsed_ticks"]),
                "reward": float(reward),
                "Y": float(next_info["Y"]),
                "type1_total": float(next_info["type1_total"]),
                "type2_total": float(next_info["type2_total"]),
                "buffer_total": float(next_info["buffer_total"]),
                "lost_orders": float(next_info["lost_orders"]),
                "current_total_setpoint": float(next_info["current_total_setpoint"]),
            }
            selected = np.asarray(next_info["selected_action_index"])
            executed = np.asarray(next_info["executed_action_index"])
            applied = np.asarray(next_info["action_applied_mask"])
            station_credit = np.asarray(
                next_info.get(
                    "station_difference_reward",
                    np.zeros(NUM_STATIONS, dtype=np.float32),
                )
            )
            for station in range(NUM_STATIONS):
                row[f"selected_action_s{station+1}"] = int(selected[station])
                row[f"executed_action_s{station+1}"] = int(executed[station])
                row[f"active_s{station+1}"] = int(applied[station])
                row[f"station_credit_s{station+1}"] = float(
                    station_credit[station]
                )
            action_rows.append(row)

            decision_index += 1
            state, info = next_state, next_info

        gate_accuracy = np.nan
        gate_alignment = np.nan
        if states_for_gate:
            gate_probabilities = model.gate_probabilities_np(
                np.asarray(states_for_gate, dtype=np.float32)
            )
            if gate_probabilities is not None:
                regime_array = np.asarray(regimes_for_gate, dtype=np.int64)
                gate_accuracy = float(
                    np.mean(np.argmax(gate_probabilities, axis=1) == regime_array)
                )
                gate_alignment = float(
                    gate_probabilities[
                        np.arange(len(regime_array)), regime_array
                    ].mean()
                )
                for regime_id, probabilities in zip(
                    regime_array, gate_probabilities
                ):
                    gate_row = {
                        "replication": int(replication_id),
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "gate_scope": model.gate_scope,
                        "scenario_index": int(scenario_index),
                        "scenario_seed": int(scenario_seed),
                        "true_regime_id": int(regime_id),
                        "true_regime": REGIME_NAMES[int(regime_id)],
                    }
                    for index, name in enumerate(REGIME_NAMES):
                        gate_row[f"gate_p_{name}"] = float(probabilities[index])
                    gate_rows.append(gate_row)

        scenario_rows.append(
            {
                "split": "validation",
                "replication": int(replication_id),
                "model": model_name,
                "model_label": model_label(model_name),
                "scenario_index": int(scenario_index),
                "scenario_seed": int(scenario_seed),
                "objective_value": float(total_reward),
                "reward": float(total_reward),
                "Y": float(info["Y"]),
                "type1_total": float(info["type1_total"]),
                "type2_total": float(info["type2_total"]),
                "buffer_total": float(info["buffer_total"]),
                "lost_orders": float(info["lost_orders"]),
                "decision_count": int(info["decision_count"]),
                "mean_elapsed_ticks": float(np.mean(elapsed_ticks)) if elapsed_ticks else 0.0,
                "gate_acc": gate_accuracy,
                "gate_regime_alignment": gate_alignment,
                "regime_normal_count": int(regime_counts[0]),
                "regime_congestion_count": int(regime_counts[1]),
                "regime_starvation_count": int(regime_counts[2]),
                "regime_quality_risk_count": int(regime_counts[3]),
            }
        )
        env.close()

    scenario_df = pd.DataFrame(scenario_rows)
    gate_df = pd.DataFrame(gate_rows)
    action_df = pd.DataFrame(action_rows)

    # Actor-MoE receives a second, paired evaluation using the true regime to
    # choose the policy expert.  Results are merged by the same scenario seed.
    oracle_actor_df = collect_actor_oracle_policy_validation(
        model, env_kwargs, validation_seeds
    )
    if not oracle_actor_df.empty:
        scenario_df = scenario_df.merge(
            oracle_actor_df, on=["scenario_index", "scenario_seed"], how="left"
        )
    else:
        for column in [
            "oracle_objective_value", "oracle_Y", "oracle_type1_total",
            "oracle_type2_total", "oracle_buffer_total",
            "oracle_lost_orders", "oracle_decision_count",
        ]:
            scenario_df[column] = np.nan

    # Use a bounded diagnostic sample to avoid excessive validation-memory use.
    diagnostic_transitions = all_transitions[-min(len(all_transitions), 4096) :]
    old_data = model.data
    model.data = diagnostic_transitions.copy()
    diagnostic_batch = model.make_batch(clear=False)
    oracle_routing_metrics = model.oracle_routing_diagnostics(diagnostic_batch)
    actor_loss, critic_loss, aux = model.eval_net_on_buffer(entropy_coeff)
    model.data = old_data

    summary: Dict[str, float] = {
        "replication": int(replication_id),
        "model": model_name,
        "model_label": model_label(model_name),
        "gate_scope": model.gate_scope,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "mean_actor_grad_norm": float(aux.get("mean_actor_grad_norm", np.nan)),
        "mean_critic_grad_norm": float(aux.get("mean_critic_grad_norm", np.nan)),
        "mean_gate_grad_norm": float(aux.get("mean_gate_grad_norm", np.nan)),
        "gate_ce": float(aux.get("gate_ce", np.nan)),
        "expert_td": float(aux.get("expert_td", np.nan)),
        "expert_correlation_loss": float(
            aux.get("expert_correlation_loss", np.nan)
        ),
        "routing_oracle_weight": float(
            aux.get("routing_oracle_weight", np.nan)
        ),
        "gate_alignment_ema": float(aux.get("gate_alignment_ema", np.nan)),
        "gate_sharpening_enabled": float(
            aux.get("gate_sharpening_enabled", np.nan)
        ),
        "setpoint_guidance_loss": float(
            aux.get("setpoint_guidance_loss", np.nan)
        ),
        "mean_abs_station_credit": float(
            aux.get("mean_abs_station_credit", np.nan)
        ),
        "mean_advantage": float(aux.get("mean_advantage", np.nan)),
        "std_advantage": float(aux.get("std_advantage", np.nan)),
        "mean_abs_advantage": float(
            aux.get("mean_abs_advantage", np.nan)
        ),
        "min_advantage": float(aux.get("min_advantage", np.nan)),
        "max_advantage": float(aux.get("max_advantage", np.nan)),
        "positive_advantage_fraction": float(
            aux.get("positive_advantage_fraction", np.nan)
        ),
        "mean_normalized_advantage": float(
            aux.get("mean_normalized_advantage", np.nan)
        ),
        "std_normalized_advantage": float(
            aux.get("std_normalized_advantage", np.nan)
        ),
        "mean_abs_normalized_advantage": float(
            aux.get("mean_abs_normalized_advantage", np.nan)
        ),
        "mean_station_advantage": float(
            aux.get("mean_station_advantage", np.nan)
        ),
        "std_station_advantage": float(
            aux.get("std_station_advantage", np.nan)
        ),
        "mean_abs_station_advantage": float(
            aux.get("mean_abs_station_advantage", np.nan)
        ),
        "gate_entropy": float(aux.get("gate_entropy", np.nan)),
        "gate_balance": float(aux.get("gate_balance", np.nan)),
        "base_value_loss": float(aux.get("base_value_loss", np.nan)),
        "residual_value_l2": float(aux.get("residual_value_l2", np.nan)),
        "mixture_value_loss_coef": float(
            aux.get("mixture_value_loss_coef", np.nan)
        ),
        "assigned_expert_loss_coef": float(
            aux.get("assigned_expert_loss_coef", np.nan)
        ),
        "head_diversity_coef": float(aux.get("head_diversity_coef", np.nan)),
        "objective_value_loss": float(
            aux.get("objective_value_loss", np.nan)
        ),
        "scalar_consistency_loss": float(
            aux.get("scalar_consistency_loss", np.nan)
        ),
        "mean_value_step_yield": float(
            aux.get("mean_value_step_yield", np.nan)
        ),
        "mean_value_step_type1": float(
            aux.get("mean_value_step_type1", np.nan)
        ),
        "mean_value_step_type2": float(
            aux.get("mean_value_step_type2", np.nan)
        ),
        "mean_value_step_buffer_change": float(
            aux.get("mean_value_step_buffer_change", np.nan)
        ),
        "mean_value_step_lost_orders": float(
            aux.get("mean_value_step_lost_orders", np.nan)
        ),
        "actor_residual_l2": float(aux.get("actor_residual_l2", np.nan)),
        "actor_residual_alpha": float(aux.get("actor_residual_alpha", np.nan)),
        "actor_routing_confidence": float(
            aux.get("actor_routing_confidence", np.nan)
        ),
        "num_validation_scenarios": int(len(scenario_df)),
    }
    summary.update(oracle_routing_metrics)
    for column in [
        "objective_value",
        "Y",
        "type1_total",
        "type2_total",
        "buffer_total",
        "lost_orders",
        "decision_count",
        "mean_elapsed_ticks",
        "gate_acc",
        "gate_regime_alignment",
    ]:
        summary[column] = float(scenario_df[column].mean())
        summary[f"{column}_within_rep_std"] = float(
            scenario_df[column].std(ddof=1)
        ) if len(scenario_df) > 1 else 0.0
    for column in [
        "oracle_objective_value",
        "oracle_Y",
        "oracle_type1_total",
        "oracle_type2_total",
        "oracle_buffer_total",
        "oracle_lost_orders",
        "oracle_decision_count",
    ]:
        values = scenario_df[column].dropna()
        summary[column] = float(values.mean()) if not values.empty else np.nan
        summary[f"{column}_within_rep_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ) if not values.empty else np.nan
    if np.isfinite(summary.get("oracle_objective_value", np.nan)):
        summary["oracle_actor_objective_gain"] = float(
            summary["oracle_objective_value"] - summary["objective_value"]
        )
    else:
        summary["oracle_actor_objective_gain"] = np.nan
    return scenario_df, gate_df, action_df, summary


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_model_then_validate(
    model_name: str,
    episodes: int,
    env_kwargs: Dict,
    results_dir: str,
    replication_id: int,
    model_seed: int,
    train_seeds: List[int],
    validation_seeds: List[int],
    args: argparse.Namespace,
):
    result_path = Path(results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_global_seed(model_seed)

    probe = ManufacturingQueueRegimeEventEnv(**env_kwargs)
    state_dim = int(probe.observation_space.shape[0])
    probe.close()
    model = instantiate_model(model_name, state_dim, args)
    parameter_counts = parameter_breakdown(model)
    validate_parameter_budget(model_name, parameter_counts)
    parameter_count = parameter_counts["num_trainable_parameters"]
    print(
        f"[rep={replication_id} | {model_label(model_name)}] "
        f"state_dim={state_dim}, actor_output=4x3, "
        f"params={parameter_count:,} "
        f"(actor={parameter_counts['actor_trainable_parameters']:,}, "
        f"critic={parameter_counts['critic_trainable_parameters']:,}), device={device}"
    )

    env = ManufacturingQueueRegimeEventEnv(**env_kwargs)
    training_rows: List[Dict] = []
    model_start = time.perf_counter()

    for episode in range(episodes):
        progress = episode / max(1, episodes - 1)
        model.set_training_progress(progress)
        entropy_coeff = entropy_coefficient(episode, episodes)
        scenario_seed = int(train_seeds[episode % len(train_seeds)])
        state, info = env.reset(seed=scenario_seed)
        done = truncated = False
        total_reward = 0.0
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        aux_meter: Dict[str, List[float]] = {}
        regime_counts = np.zeros(NUM_REGIMES, dtype=np.int64)
        elapsed_ticks: List[int] = []
        update_count = 0
        episode_start = time.perf_counter()

        while not done and not truncated:
            for _ in range(ROLLOUT_DECISIONS):
                regime_id = int(info["regime_id"])
                regime_counts[regime_id] += 1
                valid_mask = env.action_mask()
                active_mask = env.active_station_mask()
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                valid_tensor = torch.tensor(
                    valid_mask, dtype=torch.bool, device=device
                ).unsqueeze(0)
                action = model.select_training_action(
                    state_tensor,
                    valid_tensor,
                    regime_id=regime_id,
                    stochastic=True,
                )
                next_state, reward, done, truncated, next_info = env.step(action)
                model.put_data(
                    make_transition(
                        state,
                        action,
                        next_state,
                        done,
                        truncated,
                        valid_mask,
                        active_mask,
                        regime_id,
                        next_info,
                    )
                )
                total_reward += float(reward)
                elapsed_ticks.append(int(next_info["elapsed_ticks"]))
                state, info = next_state, next_info
                if done or truncated:
                    break

            if model.data:
                actor_loss, critic_loss, aux = model.train_net(entropy_coeff)
                update_count += 1
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                for key, value in aux.items():
                    aux_meter.setdefault(key, []).append(float(value))

        row = {
            "split": "training",
            "replication": int(replication_id),
            "model": model_name,
            "model_label": model_label(model_name),
            "gate_scope": model.gate_scope,
            "model_seed": int(model_seed),
            "scenario_seed": int(scenario_seed),
            "episode": int(episode + 1),
            "objective_value": float(total_reward),
            "reward": float(total_reward),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else np.nan,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else np.nan,
            "mean_actor_grad_norm": float(
                np.mean(aux_meter.get("mean_actor_grad_norm", [np.nan]))
            ),
            "mean_critic_grad_norm": float(
                np.mean(aux_meter.get("mean_critic_grad_norm", [np.nan]))
            ),
            "mean_gate_grad_norm": float(
                np.mean(aux_meter.get("mean_gate_grad_norm", [np.nan]))
            ),
            "gate_acc": float(np.mean(aux_meter.get("gate_acc", [np.nan]))),
            "gate_regime_alignment": float(
                np.mean(aux_meter.get("gate_regime_alignment", [np.nan]))
            ),
            "gate_ce": float(np.mean(aux_meter.get("gate_ce", [np.nan]))),
            "expert_td": float(np.mean(aux_meter.get("expert_td", [np.nan]))),
            "expert_correlation_loss": float(
                np.mean(aux_meter.get("expert_correlation_loss", [np.nan]))
            ),
            "routing_oracle_weight": float(
                np.mean(aux_meter.get("routing_oracle_weight", [np.nan]))
            ),
            "gate_alignment_ema": float(
                np.mean(aux_meter.get("gate_alignment_ema", [np.nan]))
            ),
            "gate_sharpening_enabled": float(
                np.mean(aux_meter.get("gate_sharpening_enabled", [np.nan]))
            ),
            "setpoint_guidance_loss": float(
                np.mean(aux_meter.get("setpoint_guidance_loss", [np.nan]))
            ),
            "mean_abs_station_credit": float(
                np.mean(aux_meter.get("mean_abs_station_credit", [np.nan]))
            ),
            "mean_advantage": float(
                np.mean(aux_meter.get("mean_advantage", [np.nan]))
            ),
            "std_advantage": float(
                np.mean(aux_meter.get("std_advantage", [np.nan]))
            ),
            "mean_abs_advantage": float(
                np.mean(aux_meter.get("mean_abs_advantage", [np.nan]))
            ),
            "min_advantage": float(
                np.min(aux_meter.get("min_advantage", [np.nan]))
            ),
            "max_advantage": float(
                np.max(aux_meter.get("max_advantage", [np.nan]))
            ),
            "positive_advantage_fraction": float(
                np.mean(
                    aux_meter.get("positive_advantage_fraction", [np.nan])
                )
            ),
            "mean_normalized_advantage": float(
                np.mean(
                    aux_meter.get("mean_normalized_advantage", [np.nan])
                )
            ),
            "std_normalized_advantage": float(
                np.mean(
                    aux_meter.get("std_normalized_advantage", [np.nan])
                )
            ),
            "mean_abs_normalized_advantage": float(
                np.mean(
                    aux_meter.get("mean_abs_normalized_advantage", [np.nan])
                )
            ),
            "mean_station_advantage": float(
                np.mean(aux_meter.get("mean_station_advantage", [np.nan]))
            ),
            "std_station_advantage": float(
                np.mean(aux_meter.get("std_station_advantage", [np.nan]))
            ),
            "mean_abs_station_advantage": float(
                np.mean(
                    aux_meter.get("mean_abs_station_advantage", [np.nan])
                )
            ),
            "gate_entropy": float(
                np.mean(aux_meter.get("gate_entropy", [np.nan]))
            ),
            "gate_balance": float(
                np.mean(aux_meter.get("gate_balance", [np.nan]))
            ),
            "gate_temperature": float(
                np.mean(aux_meter.get("gate_temperature", [np.nan]))
            ),
            "policy_entropy": float(
                np.mean(aux_meter.get("policy_entropy", [np.nan]))
            ),
            "actor_residual_l2": float(
                np.mean(aux_meter.get("actor_residual_l2", [np.nan]))
            ),
            "actor_residual_alpha": float(
                np.mean(aux_meter.get("actor_residual_alpha", [np.nan]))
            ),
            "actor_routing_confidence": float(
                np.mean(aux_meter.get("actor_routing_confidence", [np.nan]))
            ),
            "base_value_loss": float(
                np.mean(aux_meter.get("base_value_loss", [np.nan]))
            ),
            "residual_value_l2": float(
                np.mean(aux_meter.get("residual_value_l2", [np.nan]))
            ),
            "mixture_value_loss_coef": float(
                np.mean(aux_meter.get("mixture_value_loss_coef", [np.nan]))
            ),
            "assigned_expert_loss_coef": float(
                np.mean(aux_meter.get("assigned_expert_loss_coef", [np.nan]))
            ),
            "head_diversity_coef": float(
                np.mean(aux_meter.get("head_diversity_coef", [np.nan]))
            ),
            "objective_value_loss": float(
                np.mean(aux_meter.get("objective_value_loss", [np.nan]))
            ),
            "scalar_consistency_loss": float(
                np.mean(aux_meter.get("scalar_consistency_loss", [np.nan]))
            ),
            "mean_value_step_yield": float(
                np.mean(aux_meter.get("mean_value_step_yield", [np.nan]))
            ),
            "mean_value_step_type1": float(
                np.mean(aux_meter.get("mean_value_step_type1", [np.nan]))
            ),
            "mean_value_step_type2": float(
                np.mean(aux_meter.get("mean_value_step_type2", [np.nan]))
            ),
            "mean_value_step_buffer_change": float(
                np.mean(
                    aux_meter.get("mean_value_step_buffer_change", [np.nan])
                )
            ),
            "mean_value_step_lost_orders": float(
                np.mean(
                    aux_meter.get("mean_value_step_lost_orders", [np.nan])
                )
            ),
            "Y": float(info["Y"]),
            "type1_total": float(info["type1_total"]),
            "type2_total": float(info["type2_total"]),
            "buffer_total": float(info["buffer_total"]),
            "lost_orders": float(info["lost_orders"]),
            "decision_count": int(info["decision_count"]),
            "mean_elapsed_ticks": float(np.mean(elapsed_ticks)) if elapsed_ticks else 0.0,
            "regime_normal_count": int(regime_counts[0]),
            "regime_congestion_count": int(regime_counts[1]),
            "regime_starvation_count": int(regime_counts[2]),
            "regime_quality_risk_count": int(regime_counts[3]),
            "num_train_updates": int(update_count),
            "episode_time_sec": float(time.perf_counter() - episode_start),
            "cumulative_time_sec": float(time.perf_counter() - model_start),
            "num_trainable_parameters": int(parameter_count),
        }
        training_rows.append(row)
        if (episode + 1) % max(1, min(50, episodes)) == 0 or episode == episodes - 1:
            print(
                f"[rep={replication_id} | {model_label(model_name)}] "
                f"episode={episode+1:4d}, reward={total_reward:8.3f}, "
                f"Y={row['Y']:.0f}, T2={row['type2_total']:.0f}, "
                f"A={row['mean_advantage']:.3f}, "
                f"gate={row['gate_acc']:.3f}"
            )

    training_df = pd.DataFrame(training_rows)
    torch.save(model.state_dict(), result_path / f"{model_name}_trained_model.pt")

    validation_df, gate_df, action_df, validation_summary = (
        collect_validation_after_training(
            model,
            env_kwargs,
            validation_seeds,
            replication_id,
            model_name,
            entropy_coefficient(max(0, episodes - 1), episodes),
        )
    )
    elapsed = float(time.perf_counter() - model_start)
    training_df["model_total_time_sec"] = elapsed
    validation_df["model_total_time_sec"] = elapsed
    validation_summary["model_total_time_sec"] = elapsed
    validation_summary["num_trainable_parameters"] = parameter_count

    training_df.to_csv(result_path / f"{model_name}_training_log.csv", index=False)
    validation_df.to_csv(
        result_path / f"{model_name}_validation_scenarios.csv", index=False
    )
    if not gate_df.empty:
        gate_df.to_csv(
            result_path / f"{model_name}_validation_gate_states.csv", index=False
        )
    if not action_df.empty:
        action_df.to_csv(
            result_path / f"{model_name}_validation_action_log.csv", index=False
        )

    parameter_info = {
        "replication": int(replication_id),
        "model": model_name,
        "model_label": model_label(model_name),
        "gate_scope": model.gate_scope,
        "state_dim": state_dim,
        "actor_output_dim": NUM_STATIONS * NUM_STATION_ACTIONS,
        "target_trainable_parameters": TARGET_TRAINABLE_PARAMETERS,
        **parameter_counts,
    }
    return (
        training_df,
        validation_df,
        gate_df,
        action_df,
        validation_summary,
        parameter_info,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def plot_mean_std_curve(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
    rolling: int = 20,
) -> None:
    if dataframe.empty or metric not in dataframe.columns:
        return
    plt.figure(figsize=(9, 5.5))
    for model_name in MODEL_ORDER:
        model_frame = dataframe[dataframe["model"] == model_name]
        if model_frame.empty:
            continue
        curves = []
        for _, replication_frame in model_frame.groupby("replication"):
            values = replication_frame.sort_values("episode")[metric]
            curves.append(values.rolling(rolling, min_periods=1).mean().to_numpy())
        min_length = min(len(curve) for curve in curves)
        matrix = np.stack([curve[:min_length] for curve in curves], axis=0)
        if not np.isfinite(matrix).any():
            continue
        x = np.arange(1, min_length + 1)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        finite = np.isfinite(mean)
        if not finite.any():
            continue
        plt.plot(x[finite], mean[finite], label=model_label(model_name), linewidth=2)
        plt.fill_between(
            x[finite], (mean - std)[finite], (mean + std)[finite], alpha=0.15
        )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"Training {ylabel.lower()}: mean ± std across replications")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_replication_training_curve(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
    rolling: int = REPLICATION_ROLLING_WINDOW,
) -> None:
    """Plot one replication with one learning curve per model."""
    if dataframe.empty or metric not in dataframe.columns:
        return
    plt.figure(figsize=(9, 5.5))
    plotted = False
    for model_name in MODEL_ORDER:
        model_frame = dataframe[dataframe["model"] == model_name].sort_values("episode")
        if model_frame.empty:
            continue
        values = pd.to_numeric(model_frame[metric], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).any():
            continue
        smooth = values.rolling(max(1, rolling), min_periods=1).mean()
        finite = np.isfinite(smooth.to_numpy(dtype=float))
        if not finite.any():
            continue
        episodes = model_frame["episode"].to_numpy(dtype=float)
        plt.plot(
            episodes[finite],
            smooth.to_numpy(dtype=float)[finite],
            label=model_label(model_name),
            linewidth=1.8,
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    replication_id = int(dataframe["replication"].iloc[0])
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"Replication {replication_id}: {ylabel}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_replication_validation_bar(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
) -> None:
    """Plot validation mean ± scenario standard deviation for one replication."""
    if dataframe.empty or metric not in dataframe.columns:
        return
    labels: List[str] = []
    means: List[float] = []
    standard_deviations: List[float] = []
    for model_name in MODEL_ORDER:
        values = pd.to_numeric(
            dataframe.loc[dataframe["model"] == model_name, metric], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        labels.append(model_label(model_name))
        means.append(float(values.mean()))
        standard_deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    if not labels:
        return
    replication_id = int(dataframe["replication"].iloc[0])
    plt.figure(figsize=(8.5, 5.2))
    positions = np.arange(len(labels))
    plt.bar(positions, means, yerr=standard_deviations, capsize=4)
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"Replication {replication_id} validation: {ylabel}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def _plot_model_order(dataframe: pd.DataFrame, gated_only: bool = False) -> List[str]:
    """Return present models in the paper-defined order."""
    candidates = GATED_MODELS if gated_only else MODEL_ORDER
    present = set(dataframe.get("model", pd.Series(dtype=str)).astype(str).unique())
    return [name for name in candidates if name in present]


def plot_replication_validation_curve(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
    gated_only: bool = False,
) -> None:
    """Plot scenario-wise validation values for one completed replication.

    A single replication has one value per model and scenario, so no replication
    uncertainty band is drawn. The final aggregate plot adds mean ± standard
    deviation across replications.
    """
    if dataframe.empty or metric not in dataframe.columns:
        return
    model_names = _plot_model_order(dataframe, gated_only=gated_only)
    plt.figure(figsize=(9, 5.5))
    plotted = False
    for model_name in model_names:
        frame = dataframe[dataframe["model"] == model_name].sort_values("scenario_index")
        if frame.empty:
            continue
        x = pd.to_numeric(frame["scenario_index"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        plt.plot(
            x[finite], y[finite], marker="o", markersize=3.0,
            linewidth=1.7, label=model_label(model_name)
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    replication_id = int(dataframe["replication"].iloc[0])
    plt.xlabel("Validation scenario index")
    plt.ylabel(ylabel)
    plt.title(f"Replication {replication_id} validation {ylabel.lower()}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_global_validation_mean_std_curve(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
    gated_only: bool = False,
) -> None:
    """Plot scenario-wise validation mean ± std across replications."""
    if dataframe.empty or metric not in dataframe.columns:
        return
    model_names = _plot_model_order(dataframe, gated_only=gated_only)
    plt.figure(figsize=(9, 5.5))
    plotted = False
    for model_name in model_names:
        frame = dataframe[dataframe["model"] == model_name].copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        summary = (
            frame.groupby("scenario_index", as_index=False)[metric]
            .agg(mean="mean", std="std", count="count")
            .sort_values("scenario_index")
        )
        if summary.empty:
            continue
        x = summary["scenario_index"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
        if not finite.any():
            continue
        plt.plot(x[finite], mean[finite], linewidth=2.0, label=model_label(model_name))
        plt.fill_between(
            x[finite], (mean - std)[finite], (mean + std)[finite], alpha=0.16
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Validation scenario index")
    plt.ylabel(ylabel)
    plt.title(f"Validation {ylabel.lower()}: mean ± std across replications")
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def build_gate_responsibility_summary(gate_df: pd.DataFrame) -> pd.DataFrame:
    """Average expert responsibilities by true regime and model."""
    if gate_df.empty:
        return pd.DataFrame()
    probability_columns = [f"gate_p_{name}" for name in REGIME_NAMES]
    required = {"model", "true_regime_id", *probability_columns}
    if not required.issubset(gate_df.columns):
        return pd.DataFrame()
    rows: List[Dict] = []
    group_columns = ["model", "true_regime_id"]
    if "replication" in gate_df.columns:
        group_columns.insert(0, "replication")
    for keys, group in gate_df.groupby(group_columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: Dict[str, object] = {}
        if "replication" in group_columns:
            row["replication"] = int(keys[0])
            model_name = str(keys[1])
            regime_id = int(keys[2])
        else:
            model_name = str(keys[0])
            regime_id = int(keys[1])
        row.update({
            "model": model_name,
            "model_label": model_label(model_name),
            "true_regime_id": regime_id,
            "true_regime": REGIME_NAMES[regime_id],
            "count": int(len(group)),
        })
        for column in probability_columns:
            row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_gate_interpretability_diagnostics(
    validation_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    filename: Path,
    aggregate_across_replications: bool,
) -> None:
    """Create critic-side gate alignment and responsibility diagnostics.

    Actor-MoE is intentionally excluded from this figure. The panel now compares
    only Critic-MoE and MH + MoE, and uses wider subplot spacing and shorter tick
    labels to prevent overlapping text.

    For replication reports, the error bar is scenario-to-scenario variability.
    For the final report, each replication is first averaged and the error bar is
    the standard deviation of those replication-level means.
    """
    if validation_df.empty or gate_df.empty:
        return

    # Keep only critic-side gated models in this interpretability figure.
    interpretability_models = ["critic_moe", "multihead_critic_moe"]
    validation_models = set(validation_df["model"].astype(str).unique())
    recorded_gate_models = set(gate_df["model"].astype(str).unique())
    gated_models = [
        model_name
        for model_name in interpretability_models
        if model_name in validation_models and model_name in recorded_gate_models
    ]
    if not gated_models:
        return

    if aggregate_across_replications:
        alignment_source = (
            validation_df.groupby(["replication", "model"], as_index=False)[
                "gate_regime_alignment"
            ].mean()
        )
        responsibility_source = (
            build_gate_responsibility_summary(gate_df)
            .groupby(["model", "true_regime_id"], as_index=False)
            .agg(
                count=("count", "sum"),
                **{
                    f"gate_p_{name}": (f"gate_p_{name}", "mean")
                    for name in REGIME_NAMES
                },
            )
        )
        top_title = "Validation gate-regime alignment: mean ± std across replications"
        figure_title = "Gate interpretability diagnostics across replications"
    else:
        alignment_source = validation_df
        responsibility_source = build_gate_responsibility_summary(gate_df)
        replication_id = int(validation_df["replication"].iloc[0])
        top_title = (
            f"Replication {replication_id} validation gate-regime alignment: "
            "mean ± scenario std"
        )
        figure_title = f"Replication {replication_id} gate interpretability diagnostics"

    # The two-column layout is deliberately wider than the original three-column
    # layout so the regime labels, expert labels, and colorbars do not overlap.
    figure = plt.figure(figsize=(14.5, 9.5), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        len(gated_models),
        height_ratios=[1.0, 1.75],
        hspace=0.58,
        wspace=0.62,
    )
    top_axis = figure.add_subplot(grid[0, :])

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for model_name in gated_models:
        values = pd.to_numeric(
            alignment_source.loc[
                alignment_source["model"] == model_name,
                "gate_regime_alignment",
            ],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        labels.append(model_label(model_name))
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    if not labels:
        plt.close(figure)
        return

    positions = np.arange(len(labels))
    bars = top_axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        width=0.56,
    )
    top_axis.set_xticks(positions)
    top_axis.set_xticklabels(labels, fontsize=11)
    top_axis.set_ylim(0.0, 1.0)
    top_axis.set_ylabel("Gate-regime alignment", fontsize=11)
    top_axis.set_title(top_title, fontsize=13, pad=10)
    top_axis.grid(axis="y", alpha=0.20)
    top_axis.margins(x=0.12)

    for bar, value, std_value in zip(bars, means, stds):
        label_height = min(0.98, value + std_value + 0.025)
        top_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    probability_columns = [f"gate_p_{name}" for name in REGIME_NAMES]
    expert_tick_labels = [
        f"Expert {expert_id}\n{name.replace('_', '-')}"
        for expert_id, name in enumerate(REGIME_NAMES)
    ]

    for plot_index, model_name in enumerate(gated_models):
        axis = figure.add_subplot(grid[1, plot_index])
        model_frame = responsibility_source[
            responsibility_source["model"] == model_name
        ]

        matrix = np.full((NUM_REGIMES, NUM_REGIMES), np.nan, dtype=float)
        counts = np.zeros(NUM_REGIMES, dtype=int)
        for regime_id in range(NUM_REGIMES):
            row = model_frame[model_frame["true_regime_id"] == regime_id]
            if row.empty:
                continue
            for expert_id, column in enumerate(probability_columns):
                matrix[regime_id, expert_id] = float(row[column].mean())
            counts[regime_id] = int(row["count"].sum())

        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="equal",
        )

        for row_index in range(NUM_REGIMES):
            for column_index in range(NUM_REGIMES):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    continue
                text_color = "white" if value < 0.45 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="semibold",
                )

        axis.set_xticks(range(NUM_REGIMES))
        axis.set_xticklabels(
            expert_tick_labels,
            rotation=23,
            ha="right",
            fontsize=9,
        )
        axis.tick_params(axis="x", pad=8)

        axis.set_yticks(range(NUM_REGIMES))
        axis.set_yticklabels(
            [
                f"Regime {regime_id}\n{name.replace('_', '-')}\nn={counts[regime_id]}"
                for regime_id, name in enumerate(REGIME_NAMES)
            ],
            fontsize=9,
        )
        axis.tick_params(axis="y", pad=8)
        axis.set_title(
            f"{model_label(model_name)} gate responsibility",
            fontsize=12,
            pad=9,
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.065,
        )
        colorbar.ax.tick_params(labelsize=9)

    figure.suptitle(figure_title, fontsize=15, y=0.985)
    figure.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.12,
        top=0.89,
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_replication_parameter_counts(
    parameter_df: pd.DataFrame,
    filename: Path,
) -> None:
    if parameter_df.empty:
        return
    ordered = []
    for model_name in MODEL_ORDER:
        row = parameter_df[parameter_df["model"] == model_name]
        if not row.empty:
            ordered.append(row.iloc[0])
    if not ordered:
        return
    labels = [str(row["model_label"]) for row in ordered]
    totals = [float(row["num_trainable_parameters"]) for row in ordered]
    actor_counts = [float(row["actor_trainable_parameters"]) for row in ordered]
    critic_counts = [float(row["critic_trainable_parameters"]) for row in ordered]
    positions = np.arange(len(labels))
    plt.figure(figsize=(8.5, 5.2))
    plt.bar(positions, actor_counts, label="Actor")
    plt.bar(positions, critic_counts, bottom=actor_counts, label="Critic")
    plt.axhline(
        TARGET_TRAINABLE_PARAMETERS,
        linestyle="--",
        linewidth=1.2,
        label=f"Target {TARGET_TRAINABLE_PARAMETERS:,}",
    )
    for position, total in zip(positions, totals):
        plt.text(position, total + 80, f"{int(total):,}", ha="center", fontsize=8)
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel("Trainable parameters")
    plt.title("Parameter-budget comparison")
    plt.legend()
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_oracle_routing_validation(
    validation_summary_df: pd.DataFrame,
    output_path: Path,
    aggregate_across_replications: bool,
) -> None:
    """Plot critic-routing diagnostics and Actor-MoE oracle policy results."""
    if validation_summary_df.empty:
        return
    frame = validation_summary_df.copy()
    required_columns = [
        "critic_mc_huber_learned",
        "critic_mc_huber_oracle",
        "objective_value",
        "oracle_objective_value",
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    if aggregate_across_replications:
        frame = frame.groupby("model", as_index=False).agg(
            critic_mc_huber_learned=("critic_mc_huber_learned", "mean"),
            critic_mc_huber_oracle=("critic_mc_huber_oracle", "mean"),
            objective_value=("objective_value", "mean"),
            oracle_objective_value=("oracle_objective_value", "mean"),
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    critic_models = ["critic_moe", "multihead_critic_moe"]
    critic_frame = frame[frame["model"].isin(critic_models)].copy()
    critic_frame = critic_frame.dropna(
        subset=["critic_mc_huber_learned", "critic_mc_huber_oracle"],
        how="all",
    )
    if critic_frame.empty:
        axes[0].text(0.5, 0.5, "No critic-side MoE diagnostics", ha="center")
        axes[0].set_axis_off()
    else:
        x = np.arange(len(critic_frame))
        width = 0.35
        axes[0].bar(
            x - width / 2, critic_frame["critic_mc_huber_learned"],
            width, label="Learned routing"
        )
        axes[0].bar(
            x + width / 2, critic_frame["critic_mc_huber_oracle"],
            width, label="Oracle routing"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            [model_label(name) for name in critic_frame["model"]]
        )
        axes[0].set_ylabel("MC Huber value error (lower is better)")
        axes[0].set_title("Critic routing on fixed validation transitions")
        axes[0].legend()

    actor_frame = frame[frame["model"] == "actor_moe"].copy()
    actor_frame = actor_frame.dropna(
        subset=["objective_value", "oracle_objective_value"], how="any"
    )
    if actor_frame.empty:
        axes[1].text(0.5, 0.5, "No Actor-MoE oracle policy result", ha="center")
        axes[1].set_axis_off()
    else:
        learned = float(actor_frame["objective_value"].mean())
        oracle = float(actor_frame["oracle_objective_value"].mean())
        axes[1].bar([0, 1], [learned, oracle])
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Learned gate", "Oracle regime"])
        axes[1].set_ylabel("Validation objective (higher is better)")
        axes[1].set_title("Actor-MoE policy routing")
        for index, value in enumerate([learned, oracle]):
            axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    title = (
        "Oracle-routing validation across replications"
        if aggregate_across_replications
        else "Oracle-routing validation"
    )
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_validation_summary_metric_bar(
    dataframe: pd.DataFrame,
    metric: str,
    filename: Path,
    ylabel: str,
    aggregate_across_replications: bool,
) -> None:
    """Plot one validation diagnostic metric for every model.

    A replication plot shows the one diagnostic-buffer estimate per model. The
    aggregate plot shows mean ± standard deviation of those replication-level
    estimates.
    """
    if dataframe.empty or metric not in dataframe.columns:
        return
    labels: List[str] = []
    means: List[float] = []
    standard_deviations: List[float] = []
    for model_name in MODEL_ORDER:
        values = pd.to_numeric(
            dataframe.loc[dataframe["model"] == model_name, metric],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        labels.append(model_label(model_name))
        means.append(float(values.mean()))
        standard_deviations.append(
            float(values.std(ddof=1))
            if aggregate_across_replications and len(values) > 1
            else 0.0
        )
    if not labels:
        return
    plt.figure(figsize=(8.5, 5.2))
    positions = np.arange(len(labels))
    plt.bar(positions, means, yerr=standard_deviations, capsize=4)
    plt.axhline(0.0, linewidth=0.9, alpha=0.5)
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel(ylabel)
    if aggregate_across_replications:
        title = f"Validation {ylabel.lower()}: mean ± std across replications"
    else:
        replication_id = int(dataframe["replication"].iloc[0])
        title = f"Replication {replication_id} validation: {ylabel}"
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=220)
    plt.close()


def generate_replication_reports(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    action_df: pd.DataFrame,
    replication_dir: Path,
) -> None:
    """Save CSVs and both legacy and manuscript-style figures for one replication."""
    plot_dir = replication_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    training_df.to_csv(replication_dir / "all_training_logs.csv", index=False)
    validation_df.to_csv(
        replication_dir / "all_validation_scenario_results.csv", index=False
    )
    validation_summary_df.to_csv(
        replication_dir / "validation_summary_after_training.csv", index=False
    )
    parameter_df.to_csv(replication_dir / "parameter_counts.csv", index=False)
    if not gate_df.empty:
        gate_df.to_csv(replication_dir / "all_validation_gate_states.csv", index=False)
    if not action_df.empty:
        action_df.to_csv(replication_dir / "all_validation_action_log.csv", index=False)

    # Existing per-replication curves and bars are retained.
    training_specs = [
        ("objective_value", "training_objective.png", "Episode objective"),
        ("actor_loss", "training_actor_loss.png", "Actor loss"),
        ("critic_loss", "training_critic_loss.png", "Critic loss"),
        ("mean_actor_grad_norm", "training_actor_gradient_norm.png", "Actor gradient norm"),
        ("mean_critic_grad_norm", "training_critic_gradient_norm.png", "Critic gradient norm"),
        ("mean_advantage", "training_mean_advantage.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "training_mean_abs_advantage.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "training_mean_abs_station_advantage.png", "Mean absolute active-station advantage"),
        ("positive_advantage_fraction", "training_positive_advantage_fraction.png", "Positive-advantage fraction"),
        ("Y", "training_good_quality_yield.png", "Good-quality yield"),
        ("type1_total", "training_type1_defects.png", "Type-I defects"),
        ("type2_total", "training_type2_defects.png", "Type-II defects"),
        ("gate_acc", "training_gate_accuracy.png", "Gate accuracy"),
        ("gate_regime_alignment", "training_gate_alignment.png", "Gate-regime alignment"),
        ("gate_entropy", "training_gate_entropy.png", "Gate entropy"),
    ]
    for metric, name, ylabel in training_specs:
        plot_replication_training_curve(training_df, metric, plot_dir / name, ylabel)

    validation_specs = [
        ("objective_value", "validation_objective.png", "Objective value"),
        ("Y", "validation_good_quality_yield.png", "Good-quality yield"),
        ("type1_total", "validation_type1_defects.png", "Type-I defects"),
        ("type2_total", "validation_type2_defects.png", "Type-II defects"),
        ("buffer_total", "validation_buffer_total.png", "Final buffer total"),
        ("lost_orders", "validation_lost_orders.png", "Lost orders"),
        ("gate_acc", "validation_gate_accuracy.png", "Gate accuracy"),
        ("gate_regime_alignment", "validation_gate_alignment.png", "Gate-regime alignment"),
    ]
    for metric, name, ylabel in validation_specs:
        plot_replication_validation_bar(validation_df, metric, plot_dir / name, ylabel)

    # Additional graph set matching the supplied manuscript figures.
    requested_training_specs = [
        ("critic_loss", "training_critic_loss_curve.png", "Critic loss"),
        ("gate_regime_alignment", "training_gate_alignment_curve.png", "Gate-regime alignment"),
        ("objective_value", "training_objective_curve.png", "Objective value (maximize)"),
        ("mean_actor_grad_norm", "training_actor_gradient_curve.png", "Actor gradient norm"),
        ("actor_loss", "training_actor_loss_curve.png", "Actor loss"),
        ("mean_critic_grad_norm", "training_critic_gradient_curve.png", "Critic gradient norm"),
        ("mean_advantage", "training_advantage_curve.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "training_abs_advantage_curve.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "training_station_advantage_curve.png", "Mean absolute active-station advantage"),
    ]
    for metric, name, ylabel in requested_training_specs:
        plot_replication_training_curve(training_df, metric, plot_dir / name, ylabel)

    requested_validation_specs = [
        ("gate_regime_alignment", "validation_gate_alignment_curve.png", "Gate-regime alignment", True),
        ("objective_value", "validation_objective_curve.png", "Objective value (maximize)", False),
        ("Y", "validation_yield_curve.png", "Good-quality yield", False),
    ]
    for metric, name, ylabel, gated_only in requested_validation_specs:
        plot_replication_validation_curve(
            validation_df, metric, plot_dir / name, ylabel, gated_only=gated_only
        )

    plot_gate_interpretability_diagnostics(
        validation_df,
        gate_df,
        plot_dir / "gate_interpretability_diagnostics.png",
        aggregate_across_replications=False,
    )
    plot_replication_parameter_counts(parameter_df, plot_dir / "parameter_counts.png")
    plot_oracle_routing_validation(
        validation_summary_df,
        plot_dir / "oracle_routing_validation.png",
        aggregate_across_replications=False,
    )
    for metric, filename, ylabel in [
        ("mean_advantage", "validation_mean_advantage.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "validation_mean_abs_advantage.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "validation_mean_abs_station_advantage.png", "Mean absolute active-station advantage"),
    ]:
        plot_validation_summary_metric_bar(
            validation_summary_df, metric, plot_dir / filename, ylabel,
            aggregate_across_replications=False,
        )



def summarize_results(
    training_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    final_training = (
        training_df.sort_values("episode")
        .groupby(["replication", "model"], as_index=False)
        .tail(1)
    )
    training_aggregate = final_training.groupby("model").agg(
        training_objective_mean=("objective_value", "mean"),
        training_objective_std=("objective_value", "std"),
        training_yield_mean=("Y", "mean"),
        training_type2_mean=("type2_total", "mean"),
    ).reset_index()
    validation_aggregate = validation_summary_df.groupby("model").agg(
        validation_objective_mean=("objective_value", "mean"),
        validation_objective_std=("objective_value", "std"),
        validation_yield_mean=("Y", "mean"),
        validation_type1_mean=("type1_total", "mean"),
        validation_type2_mean=("type2_total", "mean"),
        validation_gate_acc_mean=("gate_acc", "mean"),
        validation_gate_alignment_mean=("gate_regime_alignment", "mean"),
    ).reset_index()
    parameter_mean = parameter_df.groupby("model", as_index=False).agg(
        num_trainable_parameters=("num_trainable_parameters", "mean"),
        actor_trainable_parameters=("actor_trainable_parameters", "mean"),
        critic_trainable_parameters=("critic_trainable_parameters", "mean"),
        parameter_difference_from_target=("parameter_difference_from_target", "mean"),
    )
    summary = training_aggregate.merge(validation_aggregate, on="model", how="outer")
    summary = summary.merge(parameter_mean, on="model", how="left")
    summary["model_label"] = summary["model"].map(MODEL_DISPLAY_NAMES)
    summary.to_csv(results_dir / "model_comparison_5models_mean_std.csv", index=False)
    final_training.to_csv(
        results_dir / "training_final_results_by_replication.csv", index=False
    )
    validation_summary_df.to_csv(
        results_dir / "validation_final_results_by_replication.csv", index=False
    )
    parameter_df.to_csv(results_dir / "parameter_counts.csv", index=False)


def main() -> None:
    global CRITIC_REWARD_SCALE
    global EXPERT_DIVERSITY_COEF
    global ORACLE_ROUTING_END_PROGRESS
    global BLENDED_ROUTING_END_PROGRESS
    global GATE_ALIGNMENT_SHARPEN_THRESHOLD
    global ACTOR_GRAD_CLIP, CRITIC_GRAD_CLIP, GATE_GRAD_CLIP
    global GATE_LR
    global ACTOR_RESIDUAL_LR_MULTIPLIER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--seed_stride", type=int, default=10000)
    parser.add_argument("--num_scenarios", type=int, default=120)
    parser.add_argument("--validation_ratio", type=float, default=0.25)
    parser.add_argument("--validation_episodes", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results_v19_equal_reward_weights_10k_basic_a2c_rep3")

    # Stabilization/ablation switches. Set a coefficient to 0 to remove that
    # mechanism without editing the source code.
    parser.add_argument(
        "--critic_reward_scale", type=float, default=CRITIC_REWARD_SCALE
    )
    parser.add_argument(
        "--expert_diversity_coef",
        type=float,
        default=EXPERT_DIVERSITY_COEF,
    )
    parser.add_argument(
        "--oracle_routing_end_progress",
        type=float,
        default=ORACLE_ROUTING_END_PROGRESS,
    )
    parser.add_argument(
        "--blended_routing_end_progress",
        type=float,
        default=BLENDED_ROUTING_END_PROGRESS,
    )
    parser.add_argument(
        "--gate_alignment_sharpen_threshold",
        type=float,
        default=GATE_ALIGNMENT_SHARPEN_THRESHOLD,
    )
    parser.add_argument(
        "--actor_grad_clip", type=float, default=ACTOR_GRAD_CLIP
    )
    parser.add_argument(
        "--critic_grad_clip", type=float, default=CRITIC_GRAD_CLIP
    )
    parser.add_argument(
        "--gate_grad_clip", type=float, default=GATE_GRAD_CLIP
    )
    parser.add_argument(
        "--gate_lr", type=float, default=GATE_LR,
        help="Learning rate for critic-side and actor-side regime gates.",
    )
    parser.add_argument(
        "--arrival_distribution",
        choices=["exponential", "erlang", "deterministic"],
        default="exponential",
    )
    parser.add_argument("--base_arrival_rate", type=float, default=0.27)
    parser.add_argument("--stock_ratio_min", type=float, default=0.65)
    parser.add_argument("--stock_ratio_max", type=float, default=1.35)
    parser.add_argument(
        "--initial_stock_ratios",
        type=float,
        nargs=4,
        default=(0.84, 0.88, 0.92, 0.96),
    )
    parser.add_argument("--process_time_cv", type=float, default=0.08)
    parser.add_argument("--min_regime_duration", type=int, default=30)
    parser.add_argument("--max_regime_duration", type=int, default=60)
    parser.add_argument("--gate_supervision_coef", type=float, default=0.50)
    parser.add_argument("--expert_specialization_coef", type=float, default=0.50)
    parser.add_argument(
        "--actor_expert_specialization_coef",
        type=float,
        default=0.0,
        help=("Deprecated in v7: the off-policy actor-expert auxiliary loss "
              "was removed; retained only for command-line compatibility."),
    )
    parser.add_argument(
        "--actor_residual_scale",
        type=float,
        default=1.5,
        help="Maximum scale of the bounded regime residual logits.",
    )
    parser.add_argument(
        "--actor_residual_l2_coef",
        type=float,
        default=0.0,
        help="L2 regularization coefficient for actor residual logits.",
    )
    parser.add_argument(
        "--actor_residual_lr_multiplier",
        type=float,
        default=ACTOR_RESIDUAL_LR_MULTIPLIER,
        help="Learning-rate multiplier for Actor-MoE residual experts.",
    )
    parser.add_argument("--gate_balance_coef", type=float, default=0.01)
    parser.add_argument("--gate_entropy_coef", type=float, default=0.001)
    parser.add_argument("--gate_temp_start", type=float, default=1.5)
    parser.add_argument("--gate_temp_end", type=float, default=0.9)
    args = parser.parse_args()

    CRITIC_REWARD_SCALE = float(args.critic_reward_scale)
    EXPERT_DIVERSITY_COEF = float(args.expert_diversity_coef)
    ORACLE_ROUTING_END_PROGRESS = float(
        args.oracle_routing_end_progress
    )
    BLENDED_ROUTING_END_PROGRESS = float(
        args.blended_routing_end_progress
    )
    GATE_ALIGNMENT_SHARPEN_THRESHOLD = float(
        args.gate_alignment_sharpen_threshold
    )
    ACTOR_GRAD_CLIP = float(args.actor_grad_clip)
    CRITIC_GRAD_CLIP = float(args.critic_grad_clip)
    GATE_GRAD_CLIP = float(args.gate_grad_clip)
    GATE_LR = float(args.gate_lr)
    ACTOR_RESIDUAL_LR_MULTIPLIER = float(args.actor_residual_lr_multiplier)

    if args.episodes < 1 or args.replications < 1:
        raise ValueError("episodes and replications must be positive.")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0,1).")
    if args.num_scenarios < 2:
        raise ValueError("num_scenarios must be at least 2.")
    if args.gate_temp_start <= 0.0 or args.gate_temp_end <= 0.0:
        raise ValueError("Gate temperatures must be positive.")
    if args.actor_residual_scale < 0.0:
        raise ValueError("actor_residual_scale must be non-negative.")
    if args.actor_residual_l2_coef < 0.0:
        raise ValueError("actor_residual_l2_coef must be non-negative.")
    if args.actor_residual_lr_multiplier <= 0.0:
        raise ValueError("actor_residual_lr_multiplier must be positive.")
    if GATE_LR <= 0.0:
        raise ValueError("gate_lr must be positive.")
    if not (
        0.0 <= ORACLE_ROUTING_END_PROGRESS
        <= BLENDED_ROUTING_END_PROGRESS
        <= 1.0
    ):
        raise ValueError(
            "Require 0 <= oracle_routing_end <= "
            "blended_routing_end <= 1."
        )
    if CRITIC_REWARD_SCALE <= 0.0:
        raise ValueError("critic_reward_scale must be positive.")
    if min(
        ACTOR_GRAD_CLIP, CRITIC_GRAD_CLIP, GATE_GRAD_CLIP
    ) <= 0.0:
        raise ValueError("Gradient clipping thresholds must be positive.")

    results_root = Path(args.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    env_kwargs = env_kwargs_from_args(args)
    selected_models = MODEL_ORDER if args.model == "all" else [args.model]

    all_training: List[pd.DataFrame] = []
    all_validation: List[pd.DataFrame] = []
    all_gate_states: List[pd.DataFrame] = []
    all_action_logs: List[pd.DataFrame] = []
    all_summaries: List[Dict] = []
    all_parameters: List[Dict] = []
    experiment_start = time.perf_counter()

    print("[EVENT_5MODEL_V19_EQUAL_WEIGHTS_10K_BASIC_A2C] Event-triggered 4x3 actor and strongly heterogeneous 27-state environment,")
    print("five-component reward decomposition with unified basic one-step A2C losses,")
    print("with regime-value-head MoE critics, MH-first MH+MoE decomposition, and ~10k budgets.")
    print(f"Device: {device}; results: {results_root}")

    for replication in range(1, args.replications + 1):
        replication_seed = args.seed + (replication - 1) * args.seed_stride
        train_seeds, validation_pool = build_scenario_splits(
            replication_seed, args.num_scenarios, args.validation_ratio
        )
        validation_seeds = (
            validation_pool
            if args.validation_episodes <= 0
            else validation_pool[: args.validation_episodes]
        )
        replication_dir = results_root / f"rep_{replication:02d}"
        replication_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"training_seed": train_seeds}).to_csv(
            replication_dir / "training_scenario_seeds.csv", index=False
        )
        pd.DataFrame({"validation_seed": validation_seeds}).to_csv(
            replication_dir / "validation_scenario_seeds.csv", index=False
        )

        replication_training: List[pd.DataFrame] = []
        replication_validation: List[pd.DataFrame] = []
        replication_summaries: List[Dict] = []
        replication_parameters: List[Dict] = []
        replication_gate_states: List[pd.DataFrame] = []
        replication_action_logs: List[pd.DataFrame] = []

        # Use one common model seed for every architecture within this replication.
        # This controls the random initialization/sampling condition consistently
        # across FC, MH, Critic-MoE, MH+MoE, and Actor-MoE. Different
        # replications still use different seeds through ``replication_seed``.
        common_model_seed = replication_seed + 1000
        pd.DataFrame(
            {
                "model": selected_models,
                "model_label": [model_label(name) for name in selected_models],
                "model_seed": [common_model_seed] * len(selected_models),
            }
        ).to_csv(replication_dir / "model_seeds.csv", index=False)

        for model_name in selected_models:
            model_seed = common_model_seed
            outputs = train_model_then_validate(
                model_name,
                args.episodes,
                env_kwargs,
                str(replication_dir),
                replication,
                model_seed,
                train_seeds,
                validation_seeds,
                args,
            )
            training_df, validation_df, gate_df, action_df, summary, parameter = outputs
            replication_training.append(training_df)
            replication_validation.append(validation_df)
            replication_summaries.append(summary)
            replication_parameters.append(parameter)
            if not gate_df.empty:
                replication_gate_states.append(gate_df)
            if not action_df.empty:
                replication_action_logs.append(action_df)
            all_training.append(training_df)
            all_validation.append(validation_df)
            if not gate_df.empty:
                all_gate_states.append(gate_df)
            if not action_df.empty:
                all_action_logs.append(action_df)
            all_summaries.append(summary)
            all_parameters.append(parameter)

        replication_training_df = pd.concat(replication_training, ignore_index=True)
        replication_validation_df = pd.concat(replication_validation, ignore_index=True)
        replication_summary_df = pd.DataFrame(replication_summaries)
        replication_parameter_df = pd.DataFrame(replication_parameters)
        replication_gate_df = (
            pd.concat(replication_gate_states, ignore_index=True)
            if replication_gate_states else pd.DataFrame()
        )
        replication_action_df = (
            pd.concat(replication_action_logs, ignore_index=True)
            if replication_action_logs else pd.DataFrame()
        )
        generate_replication_reports(
            replication_training_df,
            replication_validation_df,
            replication_summary_df,
            replication_parameter_df,
            replication_gate_df,
            replication_action_df,
            replication_dir,
        )

    training_df = pd.concat(all_training, ignore_index=True)
    validation_df = pd.concat(all_validation, ignore_index=True)
    gate_df = (
        pd.concat(all_gate_states, ignore_index=True)
        if all_gate_states
        else pd.DataFrame()
    )
    action_df = (
        pd.concat(all_action_logs, ignore_index=True)
        if all_action_logs
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(all_summaries)
    parameter_df = pd.DataFrame(all_parameters)

    training_df.to_csv(results_root / "all_training_logs.csv", index=False)
    validation_df.to_csv(
        results_root / "all_validation_scenario_results.csv", index=False
    )
    summary_df.to_csv(
        results_root / "all_validation_summaries_after_training.csv", index=False
    )
    if not gate_df.empty:
        gate_df.to_csv(results_root / "all_validation_gate_states.csv", index=False)
    if not action_df.empty:
        action_df.to_csv(
            results_root / "all_validation_action_log.csv", index=False
        )

    # Final aggregate graphs: mean ± std across completed replications.
    global_curve_specs = [
        ("objective_value", "training_objective_curve_mean_std.png", "Objective value (maximize)"),
        ("actor_loss", "training_actor_loss_curve_mean_std.png", "Actor loss"),
        ("critic_loss", "training_critic_loss_curve_mean_std.png", "Critic loss"),
        ("mean_actor_grad_norm", "training_actor_gradient_curve_mean_std.png", "Actor gradient norm"),
        ("mean_critic_grad_norm", "training_critic_gradient_curve_mean_std.png", "Critic gradient norm"),
        ("mean_advantage", "training_advantage_curve_mean_std.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "training_abs_advantage_curve_mean_std.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "training_station_advantage_curve_mean_std.png", "Mean absolute active-station advantage"),
        ("positive_advantage_fraction", "training_positive_advantage_fraction_mean_std.png", "Positive-advantage fraction"),
        ("Y", "training_yield_curve_mean_std.png", "Good-quality yield"),
        ("gate_regime_alignment", "training_gate_alignment_curve_mean_std.png", "Gate-regime alignment"),
    ]
    for metric, filename, ylabel in global_curve_specs:
        plot_mean_std_curve(training_df, metric, results_root / filename, ylabel)

    global_validation_specs = [
        ("gate_regime_alignment", "validation_gate_alignment_curve_mean_std.png", "Gate-regime alignment", True),
        ("objective_value", "validation_objective_curve_mean_std.png", "Objective value (maximize)", False),
        ("Y", "validation_yield_curve_mean_std.png", "Good-quality yield", False),
    ]
    for metric, filename, ylabel, gated_only in global_validation_specs:
        plot_global_validation_mean_std_curve(
            validation_df, metric, results_root / filename, ylabel, gated_only=gated_only
        )

    plot_gate_interpretability_diagnostics(
        validation_df,
        gate_df,
        results_root / "gate_interpretability_diagnostics_mean_std.png",
        aggregate_across_replications=True,
    )
    plot_oracle_routing_validation(
        summary_df,
        results_root / "oracle_routing_validation_mean_std.png",
        aggregate_across_replications=True,
    )
    for metric, filename, ylabel in [
        ("mean_advantage", "validation_mean_advantage_mean_std.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "validation_mean_abs_advantage_mean_std.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "validation_mean_abs_station_advantage_mean_std.png", "Mean absolute active-station advantage"),
    ]:
        plot_validation_summary_metric_bar(
            summary_df, metric, results_root / filename, ylabel,
            aggregate_across_replications=True,
        )

    # Backward-compatible aliases retained for previous analysis scripts.
    legacy_global_specs = [
        ("objective_value", "training_objective_mean_std.png", "Episode objective"),
        ("actor_loss", "training_actor_loss_mean_std.png", "Actor loss"),
        ("critic_loss", "training_critic_loss_mean_std.png", "Critic loss"),
        ("mean_actor_grad_norm", "training_actor_gradient_mean_std.png", "Actor gradient norm"),
        ("mean_critic_grad_norm", "training_critic_gradient_mean_std.png", "Critic gradient norm"),
        ("mean_advantage", "training_advantage_mean_std.png", "Mean raw SMDP-GAE advantage"),
        ("mean_abs_advantage", "training_abs_advantage_mean_std.png", "Mean absolute raw advantage"),
        ("mean_abs_station_advantage", "training_station_advantage_mean_std.png", "Mean absolute active-station advantage"),
        ("Y", "training_yield_mean_std.png", "Good-quality yield"),
        ("gate_regime_alignment", "training_gate_alignment_mean_std.png", "Gate-regime alignment"),
    ]
    for metric, filename, ylabel in legacy_global_specs:
        plot_mean_std_curve(training_df, metric, results_root / filename, ylabel)

    summarize_results(training_df, summary_df, parameter_df, results_root)

    elapsed = float(time.perf_counter() - experiment_start)
    with open(results_root / "experiment_configuration.txt", "w", encoding="utf-8") as file:
        file.write(f"device={device}\n")
        file.write("state_dim=27\n")
        file.write("actor_output=4x3_stationwise\n")
        file.write(f"gamma={GAMMA}\n")
        file.write(f"gae_lambda={GAE_LAMBDA}\n")
        file.write("advantage_logging=raw_gae,normalized_gae,active_station_advantage\n")
        file.write(f"actor_lr={ACTOR_LR}\n")
        file.write(f"critic_lr={CRITIC_LR}\n")
        file.write(f"gate_lr={GATE_LR}\n")
        file.write(f"expert_diversity_coef={EXPERT_DIVERSITY_COEF}\n")
        file.write(f"actor_residual_lr_multiplier={ACTOR_RESIDUAL_LR_MULTIPLIER}\n")
        file.write(f"target_trainable_parameters={TARGET_TRAINABLE_PARAMETERS}\n")
        file.write(f"parameter_tolerance={PARAMETER_TOLERANCE}\n")
        file.write(f"total_experiment_time_sec={elapsed:.4f}\n")
    print(f"Done. Total experiment time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
