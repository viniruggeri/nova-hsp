"""
Stommel Two-Box Thermohaline Circulation Model (D3) -- R^2.

A 2-dimensional ODE system for ocean thermohaline circulation.

State: (T, S) -- temperature and salinity difference between
polar and equatorial boxes.

ODE:
    dT/dt = eta1 - T(1 + |T - S|)
    dS/dt = eta2 - S(eta3 + |T - S|)

where:
    T = non-dimensional temperature difference
    S = non-dimensional salinity difference
    eta1 = thermal forcing (fixed)
    eta2 = freshwater forcing (stress parameter, increases)
    eta3 = relaxation time ratio (fixed)

Circulation strength: q = T - S.
    q > 0: thermally driven circulation (desired/stable mode)
    q < 0: salinity driven circulation (collapsed mode)

Bifurcation: fold -- thermal mode (T > S) disappears as eta2 increases.

Reference:
    Stommel, H. (1961). "Thermohaline convection with two stable
    regimes of flow." Tellus, 13(2), 224-230.
"""

import numpy as np
from scipy.optimize import brentq


# Fixed model parameters
ETA1 = 3.0      # Thermal forcing
ETA3 = 1.0 / 3  # Relaxation time ratio (~0.333)


def stommel_thermohaline(T: int = 500, seed: int = 42) -> tuple:
    """
    Generate trajectory of the Stommel 2-box model under increasing
    freshwater forcing.

    The system lives in R^2: state = (T_val, S_val).
    Observed variable: q = T_val - S_val (circulation strength).

    Args:
        T: Number of timesteps.
        seed: Random seed.

    Returns:
        obs_1d: np.ndarray (T,) -- circulation strength q(t) = T - S
        p_schedule: np.ndarray (T,) -- freshwater forcing eta2(t)
        bif_time: int -- timestep of fold bifurcation
        full_state: np.ndarray (T, 2) -- full state [T_val, S_val]
    """
    rng = np.random.RandomState(seed)
    dt = 0.05
    noise = 0.01

    # Stress: freshwater forcing eta2 increases
    eta2_start = 0.3
    eta2_end = 1.6
    eta2 = np.linspace(eta2_start, eta2_end, T)

    # Find bifurcation
    eta2_crit = _find_stommel_bifurcation()
    bif_t = np.argmin(np.abs(eta2 - eta2_crit))

    # State variables
    T_val = np.zeros(T)
    S_val = np.zeros(T)

    # Initialize at thermal equilibrium
    T_eq, S_eq = _thermal_equilibrium(eta2_start)
    T_val[0] = T_eq
    S_val[0] = S_eq

    for t in range(1, T):
        Tc, Sc = T_val[t - 1], S_val[t - 1]
        q = abs(Tc - Sc)
        dT = ETA1 - Tc * (1.0 + q)
        dS = eta2[t - 1] - Sc * (ETA3 + q)

        T_val[t] = Tc + dt * dT + rng.normal(0, noise)
        S_val[t] = Sc + dt * dS + rng.normal(0, noise)
        T_val[t] = max(T_val[t], 0.01)
        S_val[t] = max(S_val[t], 0.01)

    obs = T_val - S_val  # circulation strength
    full_state = np.column_stack([T_val, S_val])

    return obs, eta2, bif_t, full_state


def rollout_stommel(state0, eta2_start, eta2_end, N, H, eps=0.05, seed=0):
    """
    Monte Carlo rollout for Stommel model in R^2.

    Perturbs BOTH dimensions (T, S), integrates H steps,
    returns the observed scalar q = T - S.

    Args:
        state0: Initial state -- either (2,) array [T0, S0] or scalar q0.
        eta2_start: Freshwater forcing at start.
        eta2_end: Freshwater forcing at end.
        N: Number of MC samples.
        H: Rollout horizon.
        eps: Perturbation std (applied to each dimension).
        seed: Random seed.

    Returns:
        futures: np.ndarray (N, H) -- circulation strength q at each step.
    """
    rng = np.random.RandomState(seed)
    dt = 0.05
    noise = 0.01

    eta2_vals = np.linspace(eta2_start, eta2_end, H)

    # Parse initial state
    if np.ndim(state0) == 0:
        # Scalar -- reconstruct approximate (T, S) from q = T - S
        q0 = float(state0)
        T_eq, S_eq = _thermal_equilibrium(eta2_start)
        T_arr = np.full(N, T_eq) + rng.normal(0, eps, N)
        S_arr = np.full(N, S_eq) + rng.normal(0, eps, N)
    else:
        T_arr = np.full(N, float(state0[0])) + rng.normal(0, eps, N)
        S_arr = np.full(N, float(state0[1])) + rng.normal(0, eps, N)

    T_arr = np.clip(T_arr, 0.01, 10.0)
    S_arr = np.clip(S_arr, 0.01, 10.0)

    futures = np.zeros((N, H))
    for h in range(H):
        q = np.abs(T_arr - S_arr)
        dT = ETA1 - T_arr * (1.0 + q)
        dS = eta2_vals[h] - S_arr * (ETA3 + q)
        T_arr = T_arr + dt * dT + rng.normal(0, noise, N)
        S_arr = S_arr + dt * dS + rng.normal(0, noise, N)
        T_arr = np.clip(T_arr, 0.01, 10.0)
        S_arr = np.clip(S_arr, 0.01, 10.0)
        futures[:, h] = T_arr - S_arr  # observed: circulation strength

    return futures


def stommel_basin_stability(eta2_val, n_samples=200, n_steps=2000):
    """
    Compute basin stability (Menck et al. 2013) for the thermal mode
    at given eta2.

    Samples initial conditions uniformly in a box around the thermal
    equilibrium, integrates to convergence, counts fraction that end
    in the thermal mode (q = T - S > 0).

    Args:
        eta2_val: Freshwater forcing parameter.
        n_samples: Number of random initial conditions.
        n_steps: Integration steps to convergence.

    Returns:
        Basin stability in [0, 1].
    """
    # Check if thermal equilibrium exists
    eq = _find_stommel_equilibria(eta2_val)
    thermal = [e for e in eq if e[0] > e[1]]
    if not thermal:
        return 0.0

    T_eq, S_eq = thermal[0]

    # Sample ICs in a box around the equilibrium
    rng = np.random.RandomState(42)
    box_size = 1.5
    T_ics = T_eq + rng.uniform(-box_size, box_size, n_samples)
    S_ics = S_eq + rng.uniform(-box_size, box_size, n_samples)
    T_ics = np.clip(T_ics, 0.01, 10.0)
    S_ics = np.clip(S_ics, 0.01, 10.0)

    dt = 0.05
    T_arr = T_ics.copy()
    S_arr = S_ics.copy()

    for _ in range(n_steps):
        q = np.abs(T_arr - S_arr)
        dT = ETA1 - T_arr * (1.0 + q)
        dS = eta2_val - S_arr * (ETA3 + q)
        T_arr = T_arr + dt * dT
        S_arr = S_arr + dt * dS
        T_arr = np.clip(T_arr, 0.01, 10.0)
        S_arr = np.clip(S_arr, 0.01, 10.0)

    # Count fraction in thermal mode
    q_final = T_arr - S_arr
    return float(np.mean(q_final > 0))


def stommel_basin_width(eta2_val):
    """
    Basin width proxy for the Stommel model: circulation strength q*
    of the STABLE thermal equilibrium (the one with larger T - S).

    When two thermal equilibria coexist (near the fold), the stable
    one has larger q. This is the 2D analog of basin width: it
    decreases monotonically toward 0 at the fold bifurcation.

    Args:
        eta2_val: Freshwater forcing parameter.

    Returns:
        q* (float). Returns 0.0 if thermal mode does not exist.
    """
    eq_list = _find_stommel_equilibria(eta2_val)
    thermal = [(T_s, S_s) for T_s, S_s in eq_list if T_s > S_s]
    if thermal:
        # Take the largest q (stable equilibrium)
        q_values = [T_s - S_s for T_s, S_s in thermal]
        return max(q_values)
    return 0.0


def _thermal_equilibrium(eta2_val):
    """Find the thermal-mode equilibrium (T > S) for given eta2."""
    eq_list = _find_stommel_equilibria(eta2_val)
    thermal = [e for e in eq_list if e[0] > e[1]]
    if thermal:
        return thermal[0]
    # Fallback: approximate
    return (2.0, 0.5)


def _find_stommel_equilibria(eta2_val):
    """
    Find equilibria by solving for q = T - S in the thermal branch (q > 0)
    and haline branch (q < 0).

    In thermal mode (T > S, |T-S| = T-S = q > 0):
        T* = eta1 / (1 + q)
        S* = eta2 / (eta3 + q)
        q = eta1/(1+q) - eta2/(eta3+q)

    In haline mode (S > T, |T-S| = S-T = -q, define q_neg = S - T > 0):
        T* = eta1 / (1 + q_neg)
        S* = eta2 / (eta3 + q_neg)
        q_neg = eta2/(eta3+q_neg) - eta1/(1+q_neg)
    """
    results = []

    # Thermal branch: q > 0
    def f_thermal(q):
        return ETA1 / (1.0 + q) - eta2_val / (ETA3 + q) - q

    q_grid = np.linspace(0.01, 5.0, 1000)
    f_vals = np.array([f_thermal(q) for q in q_grid])
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i + 1] < 0:
            try:
                q_root = brentq(f_thermal, q_grid[i], q_grid[i + 1])
                T_star = ETA1 / (1.0 + q_root)
                S_star = eta2_val / (ETA3 + q_root)
                results.append((T_star, S_star))
            except ValueError:
                pass

    # Haline branch: q_neg > 0 (i.e., S > T)
    def f_haline(qn):
        return eta2_val / (ETA3 + qn) - ETA1 / (1.0 + qn) - qn

    f_vals_h = np.array([f_haline(q) for q in q_grid])
    for i in range(len(f_vals_h) - 1):
        if f_vals_h[i] * f_vals_h[i + 1] < 0:
            try:
                qn_root = brentq(f_haline, q_grid[i], q_grid[i + 1])
                T_star = ETA1 / (1.0 + qn_root)
                S_star = eta2_val / (ETA3 + qn_root)
                results.append((T_star, S_star))
            except ValueError:
                pass

    return results


def _find_stommel_bifurcation():
    """
    Find critical eta2 where the thermal mode fold bifurcation occurs.
    Sweeps eta2 and detects where the number of thermal equilibria drops.
    """
    prev_count = 0
    for eta2_test in np.linspace(0.1, 3.0, 3000):
        eq = _find_stommel_equilibria(eta2_test)
        thermal = [e for e in eq if e[0] > e[1]]
        count = len(thermal)
        if prev_count >= 2 and count < 2:
            return eta2_test
        if prev_count >= 1 and count == 0:
            return eta2_test
        prev_count = count
    return 1.0  # fallback
