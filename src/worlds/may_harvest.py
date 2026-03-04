"""
May's Harvesting Model (D4).

Classic fold bifurcation in population ecology with constant-rate harvesting.

ODE:
    dx/dt = r*x*(1 - x/K) - c

where:
    x = population density (state)
    r = intrinsic growth rate
    K = carrying capacity
    c = constant harvest rate (stress parameter, increases monotonically)

Equilibria: x* = K/2 * (1 +/- sqrt(1 - 4c/(rK)))
Bifurcation: fold at c_crit = rK/4 where the two equilibria merge.
Basin width: W(c) = x_high - x_low = K * sqrt(1 - 4c/(rK))

Reference:
    May, R. M. (1977). "Thresholds and breakpoints in ecosystems
    with a multiplicity of stable states." Nature 269, 471-477.
"""

import numpy as np


# Model parameters (module-level for consistency across functions)
_R = 1.0     # intrinsic growth rate
_K = 10.0    # carrying capacity
_C_CRIT = _R * _K / 4.0  # = 2.5, analytical bifurcation point


def may_harvesting(T: int = 500, seed: int = 42) -> tuple:
    """
    May's harvesting model with monotonically increasing harvest rate.

    dx/dt = r*x*(1 - x/K) - c

    The stress parameter 'c' (harvest rate) increases linearly from safe
    to past the fold bifurcation at c_crit = rK/4.

    Args:
        T: Number of timesteps.
        seed: Random seed for reproducibility.

    Returns:
        trajectory: np.ndarray (T,) -- population density x(t)
        p_schedule: np.ndarray (T,) -- harvest rate c(t)
        bif_time: int -- timestep of bifurcation (approx)
    """
    rng = np.random.RandomState(seed)

    dt = 0.02     # integration timestep
    noise = 0.15  # process noise (scaled to K=10)

    # Stress schedule: harvest rate increases from safe to past bifurcation
    c_start = 0.3    # well below c_crit=2.5
    c_end = 3.0      # past bifurcation
    c_schedule = np.linspace(c_start, c_end, T)

    # Bifurcation time
    bif_t = np.argmin(np.abs(c_schedule - _C_CRIT))

    # Integrate
    x = np.zeros(T)
    x[0] = _may_equilibrium_high(c_start)

    for t in range(1, T):
        c = c_schedule[t - 1]
        dxdt = _R * x[t - 1] * (1.0 - x[t - 1] / _K) - c
        x[t] = x[t - 1] + dt * dxdt + rng.normal(0, noise)
        x[t] = max(x[t], 0.01)

    return x, c_schedule, bif_t


def rollout_may(x0, c_start, c_end, N, H, eps=0.05, seed=0):
    """
    Monte Carlo rollout for May's harvesting model.

    Args:
        x0: Initial population density.
        c_start: Harvest rate at start of rollout.
        c_end: Harvest rate at end of rollout.
        N: Number of Monte Carlo samples.
        H: Rollout horizon (steps).
        eps: Perturbation std.
        seed: Random seed.

    Returns:
        futures: np.ndarray (N, H) -- trajectories.
    """
    rng = np.random.RandomState(seed)
    dt = 0.02
    noise = 0.15

    c_vals = np.linspace(c_start, c_end, H)
    x = np.full(N, float(x0)) + rng.normal(0, eps, N)
    x = np.clip(x, 0.01, _K * 1.5)
    futures = np.zeros((N, H))

    for h in range(H):
        dxdt = _R * x * (1.0 - x / _K) - c_vals[h]
        x = x + dt * dxdt + rng.normal(0, noise, N)
        x = np.clip(x, 0.01, _K * 1.5)
        futures[:, h] = x

    return futures


def may_basin_width(c_val):
    """
    Analytical basin width for May's constant-harvest model.

    W(c) = K * sqrt(1 - 4c/(rK))  for c < c_crit
    W(c) = 0                        for c >= c_crit

    This is the distance between the high and low equilibria.

    Args:
        c_val: Harvest rate parameter.

    Returns:
        Basin width (float).
    """
    disc = 1.0 - 4.0 * c_val / (_R * _K)
    if disc > 0:
        return _K * np.sqrt(disc)
    return 0.0


def _may_equilibrium_high(c):
    """High (stable) equilibrium: x* = K/2 * (1 + sqrt(1 - 4c/(rK)))."""
    disc = 1.0 - 4.0 * c / (_R * _K)
    if disc > 0:
        return _K / 2.0 * (1.0 + np.sqrt(disc))
    return _K / 2.0  # at bifurcation
