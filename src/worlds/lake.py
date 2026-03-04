"""
Scheffer Lake Eutrophication Model (D1).

Classic fold bifurcation: clear-water ↔ turbid transition.

ODE:
    dx/dt = a - b*x + r * x^p / (x^p + s^p)

where:
    x = phosphorus concentration (state)
    a = nutrient loading (stress parameter, increases monotonically)
    b = decay rate
    r = maximum recycling rate
    s = half-saturation constant
    p = Hill exponent (nonlinearity)

Bifurcation: fold at critical a_crit where stable equilibrium disappears.

Reference:
    Scheffer et al. (2001). "Catastrophic shifts in ecosystems." Nature 413.
    Scheffer (2009). "Critical Transitions in Nature and Society."
"""

import numpy as np
from scipy.optimize import brentq


def lake_eutrophication(T: int = 500, seed: int = 42) -> tuple:
    """
    Scheffer lake eutrophication model with monotonic nutrient loading.

    dx/dt = a - b*x + r * x^8 / (x^8 + s^8)

    The stress parameter 'a' (nutrient loading) increases linearly,
    driving the system through a fold bifurcation.

    Args:
        T: Number of timesteps.
        seed: Random seed for reproducibility.

    Returns:
        trajectory: np.ndarray (T,) — phosphorus concentration x(t)
        p_schedule: np.ndarray (T,) — nutrient loading a(t)
        bif_time: int — timestep of bifurcation (approx)
    """
    rng = np.random.RandomState(seed)

    # Model parameters
    b = 0.8       # decay rate
    r = 1.0       # max recycling rate
    s = 0.5       # half-saturation
    p_hill = 8    # Hill exponent (sharp switch)
    dt = 0.05     # integration timestep
    noise = 0.01  # process noise

    # Stress schedule: nutrient loading a increases from safe to critical
    a_start = 0.1   # well below bifurcation
    a_end = 0.65     # past bifurcation
    a_schedule = np.linspace(a_start, a_end, T)

    # Find bifurcation point analytically (approximate via numerical root finding)
    a_crit = _find_lake_bifurcation(b, r, s, p_hill)
    bif_t = np.argmin(np.abs(a_schedule - a_crit))

    # Integrate
    x = np.zeros(T)
    # Start at low-phosphorus equilibrium
    x[0] = _lake_equilibrium_low(a_start, b, r, s, p_hill)

    for t in range(1, T):
        a = a_schedule[t - 1]
        recycling = r * x[t - 1] ** p_hill / (x[t - 1] ** p_hill + s ** p_hill)
        dxdt = a - b * x[t - 1] + recycling
        x[t] = x[t - 1] + dt * dxdt + rng.normal(0, noise)
        x[t] = max(x[t], 0.001)  # phosphorus ≥ 0

    return x, a_schedule, bif_t


def rollout_lake(x0, a_start, a_end, N, H, eps=0.05, seed=0):
    """
    Monte Carlo rollout for lake model.

    Args:
        x0: Initial phosphorus concentration.
        a_start: Nutrient loading at start of rollout.
        a_end: Nutrient loading at end of rollout.
        N: Number of Monte Carlo samples.
        H: Rollout horizon (steps).
        eps: Perturbation std.
        seed: Random seed.

    Returns:
        futures: np.ndarray (N, H) — trajectories.
    """
    rng = np.random.RandomState(seed)
    b, r_param, s, p_hill = 0.8, 1.0, 0.5, 8
    dt, noise = 0.05, 0.01

    a_vals = np.linspace(a_start, a_end, H)
    x = np.full(N, float(x0)) + rng.normal(0, eps, N)
    x = np.clip(x, 0.001, 5.0)
    futures = np.zeros((N, H))

    for h in range(H):
        recycling = r_param * x ** p_hill / (x ** p_hill + s ** p_hill)
        dxdt = a_vals[h] - b * x + recycling
        x = x + dt * dxdt + rng.normal(0, noise, N)
        x = np.clip(x, 0.001, 5.0)
        futures[:, h] = x

    return futures


def lake_basin_width(a_val, b=0.8, r=1.0, s=0.5, p_hill=8):
    """
    Compute basin width for the lake model at given nutrient loading a.

    The basin width is the distance between the low (clear) equilibrium
    and the unstable (middle) equilibrium. When these merge at the fold
    bifurcation, the basin width goes to 0.

    Args:
        a_val: Nutrient loading parameter.
        b, r, s, p_hill: Model parameters.

    Returns:
        Basin width (float). Returns 0.0 if no bistability.
    """
    equilibria = _find_lake_equilibria(a_val, b, r, s, p_hill)
    if len(equilibria) >= 2:
        # Basin width = distance from low equilibrium to unstable middle one
        return equilibria[1] - equilibria[0]
    return 0.0


def _find_lake_equilibria(a, b, r, s, p_hill):
    """Find all equilibria of dx/dt = a - b*x + r*x^p/(x^p + s^p) = 0."""
    def f(x):
        recycling = r * x ** p_hill / (x ** p_hill + s ** p_hill)
        return a - b * x + recycling

    # Scan for sign changes
    x_grid = np.linspace(0.001, 3.0, 2000)
    f_grid = np.array([f(xi) for xi in x_grid])
    roots = []
    for i in range(len(f_grid) - 1):
        if f_grid[i] * f_grid[i + 1] < 0:
            try:
                root = brentq(f, x_grid[i], x_grid[i + 1])
                roots.append(root)
            except ValueError:
                pass
    return sorted(roots)


def _find_lake_bifurcation(b, r, s, p_hill):
    """Find critical a where fold bifurcation occurs (two equilibria merge)."""
    # Sweep a and find where number of equilibria drops from 3 to 1
    for a_test in np.linspace(0.0, 1.0, 5000):
        eqs = _find_lake_equilibria(a_test, b, r, s, p_hill)
        if len(eqs) < 2:
            return a_test
    return 0.5  # fallback


def _lake_equilibrium_low(a, b, r, s, p_hill):
    """Find the lowest (clear-water) equilibrium for given a."""
    eqs = _find_lake_equilibria(a, b, r, s, p_hill)
    if len(eqs) > 0:
        return eqs[0]
    return 0.1  # fallback
