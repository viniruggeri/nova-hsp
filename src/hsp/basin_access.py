"""
Basin Access Probability — S_t

Core metric of the HSP framework. Estimates the probability that a
perturbed state remains in the basin of attraction after H integration steps.

    S_t = P[Φ^H(x_t + ε) ∈ B(p_{t+H})]

Estimated via Monte Carlo:

    Ŝ_t = (1/N) Σ 𝟙[Φ^H(x_t + ε_i) ∈ B(p_{t+H})]

References:
    - HSP_BASIN_ACCESS.md §2.4–2.5
    - Proposition 1 (Monotonicity): NB 11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BasinAccessConfig:
    """Configuration for basin access probability estimation."""

    n_rollouts: int = 300
    """Number of Monte Carlo perturbation samples."""

    horizon: int = 80
    """Integration horizon H (steps)."""

    perturb_std: float = 0.05
    """Standard deviation of Gaussian perturbation ε."""

    step: int = 5
    """Temporal sampling resolution (compute S_t every `step` timesteps)."""

    survival_direction: Literal["above", "below"] = "above"
    """Whether survival means staying above or below threshold."""


@dataclass
class BasinAccessResult:
    """Result of basin access probability computation."""

    times: np.ndarray
    """Timestamps at which S_t was evaluated."""

    survival: np.ndarray
    """S_t values at each timestamp."""

    config: BasinAccessConfig
    """Configuration used for computation."""


def compute_basin_access(
    trajectory: np.ndarray,
    p_schedule: np.ndarray,
    rollout_fn: Callable,
    threshold: float,
    config: BasinAccessConfig | None = None,
) -> BasinAccessResult:
    """
    Compute basin access probability S_t along a trajectory.

    This is the analytical version for systems where we have direct
    access to the dynamics (f(x, p) is known). For learned dynamics,
    use `compute_basin_access_learned`.

    Args:
        trajectory: State trajectory x_t, shape (T,) or (T, d).
        p_schedule: Parameter schedule p(t), shape (T,).
        rollout_fn: Function (x0, p, H) -> x_H that integrates the dynamics.
        threshold: Survival threshold.
        config: Basin access configuration. Uses defaults if None.

    Returns:
        BasinAccessResult with times and survival probability series.
    """
    if config is None:
        config = BasinAccessConfig()

    T = len(trajectory)
    max_t = T - config.horizon
    sample_times = np.arange(0, max_t, config.step)

    survival = np.zeros(len(sample_times))

    for i, t in enumerate(sample_times):
        x_t = trajectory[t]
        p_t = p_schedule[t]

        # Monte Carlo perturbation
        perturbations = np.random.randn(config.n_rollouts) * config.perturb_std
        alive = 0

        for eps in perturbations:
            x_final = rollout_fn(x_t + eps, p_t, config.horizon)
            if config.survival_direction == "above":
                alive += int(x_final > threshold)
            else:
                alive += int(x_final < threshold)

        survival[i] = alive / config.n_rollouts

    return BasinAccessResult(
        times=sample_times.astype(float),
        survival=survival,
        config=config,
    )


def compute_basin_access_learned(
    z_sequence: torch.Tensor,
    dynamics_model: nn.Module,
    survival_fn: Callable[[torch.Tensor], torch.Tensor],
    config: BasinAccessConfig | None = None,
) -> BasinAccessResult:
    """
    Compute basin access probability using a learned dynamics model.

    For real-world data where f(x, p) is unknown. Uses a neural dynamics
    model g_θ to rollout in latent space.

    Args:
        z_sequence: Encoded latent states, shape (T, d).
        dynamics_model: Neural dynamics model g_θ: z_t -> z_{t+1}.
        survival_fn: Function that returns 1.0 if state is viable, 0.0 otherwise.
        config: Basin access configuration.

    Returns:
        BasinAccessResult with times and survival probability series.
    """
    if config is None:
        config = BasinAccessConfig()

    T, d = z_sequence.shape
    max_t = T - config.horizon
    sample_times = np.arange(0, max_t, config.step)

    survival = np.zeros(len(sample_times))

    with torch.no_grad():
        for i, t in enumerate(sample_times):
            z_t = z_sequence[t]  # (d,)

            # Perturb: z_t + ε_i for i = 1..N
            eps = torch.randn(config.n_rollouts, d) * config.perturb_std
            z_perturbed = z_t.unsqueeze(0) + eps  # (N, d)

            # Rollout H steps through learned dynamics
            z_current = z_perturbed
            for _step in range(config.horizon):
                z_current = dynamics_model(z_current)

            # Check survival
            alive = survival_fn(z_current)  # (N,) of 0/1
            survival[i] = alive.float().mean().item()

    return BasinAccessResult(
        times=sample_times.astype(float),
        survival=survival,
        config=config,
    )
