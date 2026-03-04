"""
Coupled Saddle-Node Network on R^d.

A d-dimensional ODE system of coupled saddle-node oscillators on an
all-to-all graph, demonstrating HSP S_t in arbitrary dimension.

ODE (for each node i = 1, ..., d):
    dx_i/dt = r - x_i^2 + kappa * (1/(d-1)) * sum_{j != i} (x_j - x_i)

where:
    x_i = state of node i
    r = bifurcation parameter (stress, decreases from 2 to -0.5)
    kappa = coupling strength (diffusive)

Key properties:
    - On the synchronous manifold (x_1 = ... = x_d = x), reduces to
      the scalar saddle-node: dx/dt = r - x^2.
    - Fold bifurcation at r = 0 (same as uncoupled).
    - Coupling stabilizes the synchronous state in transverse directions.
    - Basin geometry is genuinely d-dimensional.
    - For S_t: perturbation epsilon in R^d tests both the critical
      (sync) and transverse (coupling) directions.

This demonstrates that S_t works in R^n with the same algorithm:
perturb in all dimensions, rollout, count survivors.
"""

import numpy as np


def coupled_saddle_node(T: int = 400, n_dim: int = 3, seed: int = 42,
                        coupling: float = 0.1) -> tuple:
    """
    Generate trajectory of coupled saddle-node network on R^d.

    Args:
        T: Number of timesteps.
        n_dim: Dimension (number of coupled nodes).
        seed: Random seed.
        coupling: Diffusive coupling strength kappa.

    Returns:
        obs_1d: np.ndarray (T,) -- mean state (sync variable)
        p_schedule: np.ndarray (T,) -- bifurcation parameter r(t)
        bif_time: int -- timestep of bifurcation (r = 0)
        full_state: np.ndarray (T, n_dim) -- full state vector
    """
    rng = np.random.RandomState(seed)
    dt = 0.02
    noise = 0.06

    # Stress: r decreases from safe to past bifurcation
    r_start = 2.0
    r_end = -0.5
    r_schedule = np.linspace(r_start, r_end, T)
    bif_t = np.argmin(np.abs(r_schedule))  # r = 0

    # Initialize at stable equilibrium (sync state: x_i = sqrt(r))
    x = np.full(n_dim, np.sqrt(r_start))
    # Add small asymmetry so it's not perfectly synchronous
    x += rng.normal(0, 0.01, n_dim)

    full_state = np.zeros((T, n_dim))
    full_state[0] = x

    for t in range(1, T):
        r_val = r_schedule[t - 1]

        # Saddle-node dynamics + diffusive coupling
        x_mean = np.mean(x)
        coupling_term = coupling * (x_mean - x) * n_dim / max(n_dim - 1, 1)
        # Equivalent to kappa/(d-1) * sum_{j!=i}(x_j - x_i) for all-to-all

        dx = r_val - x ** 2 + coupling_term
        x = x + dt * dx + rng.normal(0, noise, n_dim)
        x = np.clip(x, -5.0, 5.0)
        full_state[t] = x

    obs = np.mean(full_state, axis=1)  # synchronous variable
    return obs, r_schedule, bif_t, full_state


def rollout_coupled_sn(state0, r_start, r_end, N, H, eps=0.05, seed=0,
                       n_dim=3, coupling=0.1):
    """
    Monte Carlo rollout for coupled saddle-node in R^d.

    Perturbs ALL d dimensions, integrates H steps, returns the
    observed mean state.

    Args:
        state0: Initial state -- (n_dim,) array or scalar (sync approx).
        r_start: Parameter r at start.
        r_end: Parameter r at end.
        N: Number of MC samples.
        H: Rollout horizon.
        eps: Perturbation std (per dimension).
        seed: Random seed.
        n_dim: Dimension.
        coupling: Coupling strength.

    Returns:
        futures: np.ndarray (N, H) -- mean state at each step.
    """
    rng = np.random.RandomState(seed)
    dt = 0.02
    noise = 0.06
    r_vals = np.linspace(r_start, r_end, H)

    # Parse initial state
    if np.ndim(state0) == 0:
        # Scalar: replicate to sync state
        x0_vec = np.full(n_dim, float(state0))
    else:
        x0_vec = np.asarray(state0, dtype=float)

    # x shape: (N, n_dim) -- N samples, each in R^d
    x = np.tile(x0_vec, (N, 1)) + rng.normal(0, eps, (N, n_dim))
    x = np.clip(x, -5.0, 5.0)

    futures = np.zeros((N, H))
    for h in range(H):
        r_val = r_vals[h]
        # Coupling: diffusive toward mean of each sample's own components
        x_mean = np.mean(x, axis=1, keepdims=True)  # (N, 1)
        coupling_term = coupling * (x_mean - x) * n_dim / max(n_dim - 1, 1)

        dx = r_val - x ** 2 + coupling_term
        x = x + dt * dx + rng.normal(0, noise, (N, n_dim))
        x = np.clip(x, -5.0, 5.0)
        futures[:, h] = np.mean(x, axis=1)  # observed: mean state

    return futures


def coupled_sn_basin_width(r_val):
    """
    Basin width proxy for the coupled saddle-node.

    On the synchronous manifold, the basin width is 2*sqrt(r)
    (same as the scalar saddle-node). The coupling stabilizes
    transverse perturbations, so this is the critical dimension.

    Args:
        r_val: Bifurcation parameter.

    Returns:
        Basin width (float).
    """
    return 2.0 * np.sqrt(max(r_val, 0.0))
