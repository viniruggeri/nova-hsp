"""
HSP Basin Access Probability — End-to-End Evaluation on All Datasets.

Runs S_t computation on 8 systems spanning R^1 to R^6:

  R^1 systems (original):
    - Saddle-Node (NB 11 canonical)
    - Double-Well (NB 11 canonical)
    - Ecosystem (NB 11 canonical)
    - Lake Eutrophication (Scheffer D1)
    - May's Harvesting (D4)

  R^n systems (new — extension for publication):
    - Stommel Thermohaline (D3, R^2)
    - Coupled Saddle-Node d=3 (R^3)
    - Coupled Saddle-Node d=6 (R^6)

Compares S_t against EWS baselines (rolling variance, AC1, skewness)
using geometric metrics from src/hsp/metrics.py.

Output:
  - Console summary table
  - Plots saved to results/hsp_evaluation/
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd

# ──────────────────────────────────────────────────────────────
# HSP PARAMETERS (from NB 11 / docs)
# ──────────────────────────────────────────────────────────────
N_ROLLOUTS = 300
HORIZON = 80
PERTURB_STD = 0.05
STEP = 5
N_SEEDS = 10  # fewer seeds for faster execution

# EWS parameters
EWS_WINDOW = 25  # rolling window for EWS statistics

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "hsp_evaluation")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. SYSTEM DEFINITIONS
# ══════════════════════════════════════════════════════════════

# ── Saddle-Node ──────────────────────────────────────────────
def saddle_node(T=300, seed=42):
    rng = np.random.RandomState(seed)
    dt, noise = 0.02, 0.08
    r = np.linspace(2.0, -0.5, T)
    x = np.zeros(T)
    x[0] = np.sqrt(r[0])
    for t in range(1, T):
        x[t] = x[t - 1] + dt * (r[t - 1] - x[t - 1] ** 2) + rng.normal(0, noise)
        x[t] = max(x[t], -5.0)
    bif_t = np.argmin(np.abs(r))
    return x, r, bif_t


def rollout_sn(x0, r_start, r_end, N, H, eps=0.05, seed=0):
    rng = np.random.RandomState(seed)
    dt, noise = 0.02, 0.08
    r_vals = np.linspace(r_start, r_end, H)
    x = np.full(N, float(x0)) + rng.normal(0, eps, N)
    futures = np.zeros((N, H))
    for h in range(H):
        x = x + dt * (r_vals[h] - x**2) + rng.normal(0, noise, N)
        x = np.clip(x, -5.0, 5.0)
        futures[:, h] = x
    return futures


def sn_basin_width(r_val):
    return 2.0 * np.sqrt(max(r_val, 0.0))


# ── Double-Well ──────────────────────────────────────────────
def double_well(T=400, seed=42):
    rng = np.random.RandomState(seed)
    dt, noise = 0.01, 0.12
    r = np.linspace(-0.3, 0.6, T)
    x = np.zeros(T)
    x[0] = -0.9
    for t in range(1, T):
        x[t] = x[t - 1] + dt * (x[t - 1] - x[t - 1] ** 3 + r[t - 1]) + rng.normal(0, noise)
        x[t] = np.clip(x[t], -3.0, 3.0)
    bif_r = 2.0 / (3.0 * np.sqrt(3.0))
    bif_t = np.argmin(np.abs(r - bif_r))
    return x, r, bif_t


def rollout_dw(x0, r_start, r_end, N, H, eps=0.05, seed=0):
    rng = np.random.RandomState(seed)
    dt, noise = 0.01, 0.12
    r_vals = np.linspace(r_start, r_end, H)
    x = np.full(N, float(x0)) + rng.normal(0, eps, N)
    futures = np.zeros((N, H))
    for h in range(H):
        x = x + dt * (x - x**3 + r_vals[h]) + rng.normal(0, noise, N)
        x = np.clip(x, -3.0, 3.0)
        futures[:, h] = x
    return futures


def dw_basin_width(r_val):
    coeffs = [1, 0, -1, -r_val]
    roots = np.roots(coeffs)
    real_roots = sorted([rt.real for rt in roots if abs(rt.imag) < 1e-8])
    if len(real_roots) == 3:
        return real_roots[1] - real_roots[0]
    return 0.0


# ── Ecosystem ────────────────────────────────────────────────
def ecosystem_shift(T=400, seed=42):
    rng = np.random.RandomState(seed)
    dt, noise, r_g, K, s = 0.05, 0.02, 1.0, 1.0, 0.2
    h = np.linspace(0.05, 0.35, T)
    x = np.zeros(T)
    x[0] = 0.8
    for t in range(1, T):
        grazing = h[t - 1] * x[t - 1] ** 2 / (x[t - 1] ** 2 + s**2)
        growth = r_g * x[t - 1] * (1.0 - x[t - 1] / K)
        x[t] = x[t - 1] + dt * (growth - grazing) + rng.normal(0, noise)
        x[t] = max(x[t], 0.01)
    bif_t = T
    for t in range(50, T):
        if x[t] < 0.3:
            bif_t = t
            break
    return x, h, bif_t


def rollout_eco(x0, h_start, h_end, N, H, eps=0.05, seed=0):
    rng = np.random.RandomState(seed)
    dt, noise, r_g, K, s = 0.05, 0.02, 1.0, 1.0, 0.2
    h_vals = np.linspace(h_start, h_end, H)
    x = np.full(N, float(x0)) + rng.normal(0, eps, N)
    x = np.clip(x, 0.01, 3.0)
    futures = np.zeros((N, H))
    for h_step in range(H):
        grazing = h_vals[h_step] * x**2 / (x**2 + s**2)
        growth = r_g * x * (1.0 - x / K)
        x = x + dt * (growth - grazing) + rng.normal(0, noise, N)
        x = np.clip(x, 0.01, 3.0)
        futures[:, h_step] = x
    return futures


def eco_basin_width(h_val, r_g=1.0, K=1.0, s=0.2):
    coeffs = [1, -1, s**2 + h_val, -(s**2)]
    roots = np.roots(coeffs)
    real_pos = sorted([rt.real for rt in roots if abs(rt.imag) < 1e-8 and rt.real > 0.01])
    if len(real_pos) >= 2:
        return real_pos[-1] - real_pos[-2]
    elif len(real_pos) == 1:
        return real_pos[0]
    return 0.0


# ── Lake Eutrophication (NEW — Scheffer D1) ─────────────────
from src.worlds.lake import lake_eutrophication, rollout_lake, lake_basin_width

# ── May's Harvesting (NEW — D4) ─────────────────────────────
from src.worlds.may_harvest import may_harvesting, rollout_may, may_basin_width

# ── Stommel Thermohaline (D3 — R^2) ─────────────────────────
from src.worlds.stommel import (
    stommel_thermohaline, rollout_stommel, stommel_basin_width,
)

# ── Coupled Saddle-Node Network (R^d) ────────────────────────
from src.worlds.coupled_sn import (
    coupled_saddle_node, rollout_coupled_sn, coupled_sn_basin_width,
)

# Wrapper generators for coupled SN at specific dimensions
def coupled_sn_d3(T=400, seed=42):
    return coupled_saddle_node(T=T, n_dim=3, seed=seed, coupling=0.1)

def coupled_sn_d6(T=400, seed=42):
    return coupled_saddle_node(T=T, n_dim=6, seed=seed, coupling=0.1)

# Wrapper rollouts that carry n_dim
def rollout_csn3(state0, r_start, r_end, N, H, eps=0.05, seed=0):
    return rollout_coupled_sn(state0, r_start, r_end, N, H, eps, seed, n_dim=3, coupling=0.1)

def rollout_csn6(state0, r_start, r_end, N, H, eps=0.05, seed=0):
    return rollout_coupled_sn(state0, r_start, r_end, N, H, eps, seed, n_dim=6, coupling=0.1)


# ══════════════════════════════════════════════════════════════
# 2. S_t COMPUTATION (replicates NB 11 compute_survival_timeseries)
# ══════════════════════════════════════════════════════════════

def compute_survival_timeseries(
    trajectory, param_schedule, rollout_fn, threshold,
    n_rollouts=N_ROLLOUTS, horizon=HORIZON, perturb_std=PERTURB_STD,
    step=STEP, survival_direction="above",
):
    """Compute S_t along a trajectory via Monte Carlo rollouts.

    trajectory can be 1D (scalar state) or 2D (T, n_dim) for R^n systems.
    The rollout_fn receives x0 which is scalar or vector accordingly.
    """
    T = trajectory.shape[0] if hasattr(trajectory, 'shape') and trajectory.ndim > 1 else len(trajectory)
    times = np.arange(0, T, step)
    survival = np.zeros(len(times))
    for i, t in enumerate(times):
        x0 = trajectory[t]
        p_start = param_schedule[t]
        p_end = param_schedule[min(t + horizon, T - 1)]
        futures = rollout_fn(x0, p_start, p_end, n_rollouts, horizon, eps=perturb_std, seed=i)
        endpoints = futures[:, -1]
        if survival_direction == "above":
            survival[i] = np.mean(endpoints > threshold)
        else:
            survival[i] = np.mean(endpoints < threshold)
    return times, survival


# ══════════════════════════════════════════════════════════════
# 3. EWS BASELINES (rolling time series — NOT binary alerts)
# ══════════════════════════════════════════════════════════════

def compute_rolling_variance(x, window=EWS_WINDOW):
    """Rolling variance (B1). Higher near tipping point (CSD)."""
    result = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        result[i] = np.var(x[i - window:i])
    return result


def compute_rolling_ac1(x, window=EWS_WINDOW):
    """Rolling lag-1 autocorrelation (B2). Higher near tipping point (CSD)."""
    result = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        seg = x[i - window:i]
        seg = seg - seg.mean()
        c0 = np.dot(seg, seg) / len(seg)
        if c0 == 0:
            result[i] = 0.0
            continue
        c1 = np.dot(seg[:-1], seg[1:]) / len(seg)
        result[i] = c1 / c0
    return result


def compute_rolling_skewness(x, window=EWS_WINDOW):
    """Rolling skewness (B3). Changes sign near bifurcation."""
    from scipy.stats import skew
    result = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        result[i] = skew(x[i - window:i])
    return result


def _partial_correlation_vs_ews(survival, tau, ews_dict):
    """Partial Spearman rho(S_t, tau | EWS). Inlined to avoid torch dependency."""
    from scipy.stats import rankdata
    n = len(survival)
    if n < 10:
        return float("nan")
    rank_s = rankdata(survival)
    rank_tau = rankdata(tau)
    ews_names = sorted(ews_dict.keys())
    if len(ews_names) == 0:
        rho, _ = spearmanr(survival, tau)
        return float(rho)
    ews_matrix = np.column_stack([rankdata(ews_dict[k]) for k in ews_names])
    X = np.column_stack([ews_matrix, np.ones(n)])
    try:
        beta_s = np.linalg.lstsq(X, rank_s, rcond=None)[0]
        beta_tau = np.linalg.lstsq(X, rank_tau, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    resid_s = rank_s - X @ beta_s
    resid_tau = rank_tau - X @ beta_tau
    rho, _ = spearmanr(resid_s, resid_tau)
    return float(rho)


def resample_ews_to_st_times(ews_full, times_st):
    """Resample EWS time series to match S_t sampling times."""
    valid = []
    for t in times_st:
        t_int = int(t)
        if t_int < len(ews_full) and not np.isnan(ews_full[t_int]):
            valid.append(ews_full[t_int])
        else:
            valid.append(np.nan)
    return np.array(valid)


# ══════════════════════════════════════════════════════════════
# 4. SYSTEM REGISTRY
# ══════════════════════════════════════════════════════════════

# Thresholds for survival check
THRESH_SN = -0.5
THRESH_DW = 0.0
THRESH_ECO = 0.3
THRESH_LAKE = 0.15  # low-phosphorus survival: x stays below threshold means "in clear state"
THRESH_MAY = 2.0    # high-population survival (x > 2 means not collapsed, K=10)
THRESH_STOMMEL = 0.0  # thermal mode: q = T - S > 0
THRESH_CSN = -0.5     # same as scalar saddle-node

SYSTEMS = [
    {
        "name": "Saddle-Node (R1)",
        "gen": saddle_node,
        "roll": rollout_sn,
        "thresh": THRESH_SN,
        "surv_dir": "above",
        "basin_fn": sn_basin_width,
        "dim": 1,
    },
    {
        "name": "Double-Well (R1)",
        "gen": double_well,
        "roll": rollout_dw,
        "thresh": THRESH_DW,
        "surv_dir": "below",
        "basin_fn": dw_basin_width,
        "dim": 1,
    },
    {
        "name": "Ecosystem (R1)",
        "gen": ecosystem_shift,
        "roll": rollout_eco,
        "thresh": THRESH_ECO,
        "surv_dir": "above",
        "basin_fn": eco_basin_width,
        "dim": 1,
    },
    {
        "name": "Lake (R1)",
        "gen": lake_eutrophication,
        "roll": rollout_lake,
        "thresh": THRESH_LAKE,
        "surv_dir": "below",
        "basin_fn": lake_basin_width,
        "dim": 1,
    },
    {
        "name": "May Harvest (R1)",
        "gen": may_harvesting,
        "roll": rollout_may,
        "thresh": THRESH_MAY,
        "surv_dir": "above",
        "basin_fn": may_basin_width,
        "perturb_std": 0.5,
        "dim": 1,
    },
    # ── R^n SYSTEMS ──────────────────────────────────────────
    {
        "name": "Stommel (R2)",
        "gen": stommel_thermohaline,
        "roll": rollout_stommel,
        "thresh": THRESH_STOMMEL,
        "surv_dir": "above",  # survival = thermal mode (q > 0)
        "basin_fn": stommel_basin_width,
        "perturb_std": 0.5,
        "dim": 2,
    },
    {
        "name": "Coupled SN (R3)",
        "gen": coupled_sn_d3,
        "roll": rollout_csn3,
        "thresh": THRESH_CSN,
        "surv_dir": "above",
        "basin_fn": coupled_sn_basin_width,
        "perturb_std": 0.15,
        "dim": 3,
    },
    {
        "name": "Coupled SN (R6)",
        "gen": coupled_sn_d6,
        "roll": rollout_csn6,
        "thresh": THRESH_CSN,
        "surv_dir": "above",
        "basin_fn": coupled_sn_basin_width,
        "perturb_std": 0.2,
        "dim": 6,
    },
]


# ══════════════════════════════════════════════════════════════
# 5. MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

def evaluate_system(sys_dict, n_seeds=N_SEEDS):
    """Run full HSP + EWS evaluation on one system across seeds."""
    name = sys_dict["name"]
    p_std = sys_dict.get("perturb_std", PERTURB_STD)
    dim = sys_dict.get("dim", 1)
    results = {
        "rho_S_W": [], "mono_frac": [],
        "rho_var_W": [], "rho_ac1_W": [], "rho_skew_W": [],
        "partial_corr": [],
    }

    for seed in range(n_seeds):
        gen_result = sys_dict["gen"](seed=seed)
        if len(gen_result) == 4:
            # Multi-dimensional system: (obs_1d, p_sched, bif_t, full_state)
            obs, p_sched, bif_t, full_state = gen_result
            state_for_st = full_state  # (T, d) — pass full state to S_t
        else:
            # 1D system: (trajectory, p_sched, bif_t)
            obs, p_sched, bif_t = gen_result
            state_for_st = obs  # scalar trajectory

        T = len(obs)

        # ── Compute S_t (using full state for R^n) ────────────
        times, surv = compute_survival_timeseries(
            state_for_st, p_sched, sys_dict["roll"], sys_dict["thresh"],
            survival_direction=sys_dict["surv_dir"],
            perturb_std=p_std,
        )

        # ── Basin width at S_t times ─────────────────────────
        basin_w = np.array([sys_dict["basin_fn"](p_sched[int(t)]) for t in times])
        valid = basin_w > 0.05

        if valid.sum() < 6:
            continue

        s_v = surv[valid]
        w_v = basin_w[valid]
        t_v = times[valid]

        # ── S_t metrics ──────────────────────────────────────
        rho_sw, _ = spearmanr(s_v, w_v)
        results["rho_S_W"].append(rho_sw)

        ds = np.diff(s_v)
        results["mono_frac"].append(np.mean(ds <= 0))

        # ── EWS baselines (full time series) — always on 1D obs ──
        var_full = compute_rolling_variance(obs)
        ac1_full = compute_rolling_ac1(obs)
        skew_full = compute_rolling_skewness(obs)

        var_ts = resample_ews_to_st_times(var_full, times)
        ac1_ts = resample_ews_to_st_times(ac1_full, times)
        skew_ts = resample_ews_to_st_times(skew_full, times)

        # EWS at valid (pre-collapse) points
        var_v = var_ts[valid]
        ac1_v = ac1_ts[valid]
        skew_v = skew_ts[valid]

        # Handle NaNs in EWS
        ews_valid = ~(np.isnan(var_v) | np.isnan(ac1_v) | np.isnan(skew_v))
        if ews_valid.sum() < 6:
            results["rho_var_W"].append(np.nan)
            results["rho_ac1_W"].append(np.nan)
            results["rho_skew_W"].append(np.nan)
            results["partial_corr"].append(np.nan)
            continue

        # Correlation of EWS with basin width
        # Note: EWS go UP near tipping, W goes DOWN → expect NEGATIVE correlation
        rho_var, _ = spearmanr(var_v[ews_valid], w_v[ews_valid])
        rho_ac1, _ = spearmanr(ac1_v[ews_valid], w_v[ews_valid])
        rho_skew, _ = spearmanr(skew_v[ews_valid], w_v[ews_valid])
        results["rho_var_W"].append(rho_var)
        results["rho_ac1_W"].append(rho_ac1)
        results["rho_skew_W"].append(rho_skew)

        # ── Partial correlation ρ(S, τ | EWS) ───────────────
        tau_v = bif_t - t_v
        ews_dict_v = {}
        if ews_valid.sum() >= 10:
            ews_dict_v["variance"] = var_v[ews_valid]
            ews_dict_v["ac1"] = ac1_v[ews_valid]
            ews_dict_v["skewness"] = skew_v[ews_valid]

            pc = _partial_correlation_vs_ews(
                s_v[ews_valid], tau_v[ews_valid], ews_dict_v
            )
            results["partial_corr"].append(pc)
        else:
            results["partial_corr"].append(np.nan)

    return results


def run_all():
    """Run evaluation on all systems and print summary."""
    print("=" * 80)
    print("  HSP Basin Access Probability -- Full Evaluation (R^1 to R^6)")
    print("  R^1: Saddle-Node, Double-Well, Ecosystem, Lake, May")
    print("  R^n: Stommel (R2), Coupled SN (R3), Coupled SN (R6)")
    print("  Baselines: Rolling Variance (B1), Rolling AC1 (B2), Rolling Skewness (B3)")
    print(f"  Parameters: N={N_ROLLOUTS}, H={HORIZON}, sigma={PERTURB_STD}, step={STEP}")
    print(f"  Seeds: {N_SEEDS}")
    print("=" * 80)

    all_results = {}
    for sys_dict in SYSTEMS:
        name = sys_dict["name"]
        print(f"\n{'-' * 60}")
        print(f"  Running: {name}")
        print(f"{'-' * 60}")

        res = evaluate_system(sys_dict, n_seeds=N_SEEDS)
        all_results[name] = res

        # Print per-system results
        def fmt(arr):
            arr = [x for x in arr if not np.isnan(x)]
            if len(arr) == 0:
                return "  N/A"
            return f"{np.mean(arr):+.3f} +/- {np.std(arr):.3f}"

        print(f"  rho(S_t, W_t):        {fmt(res['rho_S_W'])}")
        print(f"  Monotonicity:       {fmt(res['mono_frac'])}")
        print(f"  rho(Var, W_t):        {fmt(res['rho_var_W'])}")
        print(f"  rho(AC1, W_t):        {fmt(res['rho_ac1_W'])}")
        print(f"  rho(Skew, W_t):       {fmt(res['rho_skew_W'])}")
        print(f"  rho_partial(S,t|EWS): {fmt(res['partial_corr'])}")

    # ── Summary Table ────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("  SUMMARY TABLE -- Basin Contraction Correlation rho(indicator, W_t)")
    print("  R^1 systems: well-established   |   R^n systems: NEW for publication")
    print("=" * 110)
    header = f"{'System':<22} {'Dim':>4} {'S_t':>12} {'Variance':>12} {'AC1':>12} {'Skewness':>12} {'rho_part':>12}"
    print(header)
    print("-" * 110)
    for sys_d, (name, res) in zip(SYSTEMS, all_results.items()):
        def mn(arr):
            arr = [x for x in arr if not np.isnan(x)]
            if not arr:
                return "N/A"
            return f"{np.mean(arr):+.3f}"
        dim = sys_d.get("dim", 1)
        row = (
            f"{name:<22} "
            f"{'R'+str(dim):>4} "
            f"{mn(res['rho_S_W']):>12} "
            f"{mn(res['rho_var_W']):>12} "
            f"{mn(res['rho_ac1_W']):>12} "
            f"{mn(res['rho_skew_W']):>12} "
            f"{mn(res['partial_corr']):>12}"
        )
        print(row)
    print("=" * 110)
    print("\nPositive rho(S_t, W_t) = S_t tracks basin width (good).")
    print("Negative rho(EWS, W_t) = EWS goes UP when basin shrinks (expected for CSD).")
    print("rho_partial > 0 = S_t adds geometric info beyond classical EWS.\n")

    # ── Visualization: S_t vs EWS on each system (seed=0) ───
    _plot_all_systems(all_results)

    return all_results


def _plot_all_systems(all_results):
    """Create visualization panels for all systems."""
    n_sys = len(SYSTEMS)
    fig, axes = plt.subplots(3, n_sys, figsize=(4 * n_sys, 12))

    for idx, sys_dict in enumerate(SYSTEMS):
        name = sys_dict["name"]
        gen_result = sys_dict["gen"](seed=0)
        if len(gen_result) == 4:
            obs, p_sched, bif_t, full_state = gen_result
            state_for_st = full_state
        else:
            obs, p_sched, bif_t = gen_result
            state_for_st = obs
        T = len(obs)

        # Compute S_t
        p_std = sys_dict.get("perturb_std", PERTURB_STD)
        times, surv = compute_survival_timeseries(
            state_for_st, p_sched, sys_dict["roll"], sys_dict["thresh"],
            survival_direction=sys_dict["surv_dir"],
            perturb_std=p_std,
        )
        basin_w = np.array([sys_dict["basin_fn"](p_sched[int(t)]) for t in times])

        # EWS (always on 1D observed variable)
        var_full = compute_rolling_variance(obs)
        ac1_full = compute_rolling_ac1(obs)

        # ── Row 1: Trajectory + bifurcation ──────────────────
        ax = axes[0, idx]
        ax.plot(obs, "k-", lw=0.8, alpha=0.8)
        if bif_t < T:
            ax.axvline(bif_t, color="red", ls=":", alpha=0.6, label=f"bif t={bif_t}")
        dim_label = sys_dict.get("dim", 1)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Obs. x(t)")
        ax.legend(fontsize=8)

        # ── Row 2: S_t vs Basin Width ────────────────────────
        ax = axes[1, idx]
        valid = basin_w > 0.05
        ax.plot(times[valid], surv[valid], "b-", lw=2, label="$S_t$")
        ax2 = ax.twinx()
        ax2.plot(times[valid], basin_w[valid], "g--", lw=1.5, alpha=0.7, label="$W_t$")
        ax.set_ylabel("$S_t$ (blue)", color="blue")
        ax2.set_ylabel("Basin Width $W_t$ (green)", color="green")
        if len([x for x in all_results[name]["rho_S_W"] if not np.isnan(x)]) > 0:
            rho = np.nanmean(all_results[name]["rho_S_W"])
            ax.set_title(f"rho(S,W) = {rho:+.3f}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        # ── Row 3: EWS comparison ────────────────────────────
        ax = axes[2, idx]
        t_range = np.arange(len(obs))
        valid_var = ~np.isnan(var_full)
        valid_ac1 = ~np.isnan(ac1_full)
        if valid_var.any():
            # Normalize for visual comparison
            vn = var_full[valid_var]
            vn = (vn - vn.min()) / (vn.max() - vn.min() + 1e-10)
            ax.plot(t_range[valid_var], vn, "r-", lw=1, alpha=0.7, label="Variance (norm)")
        if valid_ac1.any():
            an = ac1_full[valid_ac1]
            an = (an - an.min()) / (an.max() - an.min() + 1e-10)
            ax.plot(t_range[valid_ac1], an, "orange", lw=1, alpha=0.7, label="AC1 (norm)")
        # Overlay normalized S_t
        s_norm = (surv - surv.min()) / (surv.max() - surv.min() + 1e-10)
        ax.plot(times, s_norm, "b-", lw=2, label="$S_t$ (norm)")
        if bif_t < T:
            ax.axvline(bif_t, color="red", ls=":", alpha=0.4)
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized indicator")
        ax.legend(fontsize=6)
        ax.set_title("$S_t$ vs EWS", fontsize=9)

    plt.suptitle(
        "HSP Basin Access Probability — R^1 to R^6 Benchmark",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hsp_full_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    run_all()
