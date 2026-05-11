"""Run analytical S_t experiments on synthetic systems.

This script operationalizes Sprint 1 from docs/ROADMAP.md:
- 10 seeds x 3 systems x 3 sigma values
- geometric metrics: rho(S_t, W_t), monotonicity fraction, normalized lead time
- results exported under results/synthetic/

Run:
    python -m src.experiments.run_hsp_synthetic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

from src.hsp.metrics import basin_contraction_correlation, monotonicity_fraction


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman rho, handling constant-input edge cases as NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho = spearmanr(x, y)[0]
    return float(np.asarray(rho).item())


@dataclass
class SystemSpec:
    name: str
    generator: Callable[..., tuple[np.ndarray, np.ndarray, int]]
    n_steps: int
    rollout: Callable[[float, float, float, int, int, float, int], np.ndarray]
    basin_width: Callable[[float], float]
    threshold: float
    survival_direction: str


def saddle_node(T: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.RandomState(seed)
    dt, noise = 0.02, 0.08
    r = np.linspace(2.0, -0.5, T)
    x = np.zeros(T)
    x[0] = np.sqrt(r[0])
    for t in range(1, T):
        x[t] = x[t - 1] + dt * (r[t - 1] - x[t - 1] ** 2) + rng.normal(0, noise)
        x[t] = max(x[t], -5.0)
    bif_t = int(np.argmin(np.abs(r)))
    return x, r, bif_t


def rollout_sn(
    x0: float,
    r_start: float,
    r_end: float,
    N: int,
    H: int,
    eps: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
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


def sn_basin_width(r_val: float) -> float:
    return 2.0 * np.sqrt(max(r_val, 0.0))


def double_well(T: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.RandomState(seed)
    dt, noise = 0.01, 0.12
    r = np.linspace(-0.3, 0.6, T)
    x = np.zeros(T)
    x[0] = -0.9
    for t in range(1, T):
        x[t] = x[t - 1] + dt * (x[t - 1] - x[t - 1] ** 3 + r[t - 1]) + rng.normal(0, noise)
        x[t] = np.clip(x[t], -3.0, 3.0)
    bif_r = 2.0 / (3.0 * np.sqrt(3.0))
    bif_t = int(np.argmin(np.abs(r - bif_r)))
    return x, r, bif_t


def rollout_dw(
    x0: float,
    r_start: float,
    r_end: float,
    N: int,
    H: int,
    eps: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
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


def dw_basin_width(r_val: float) -> float:
    coeffs = [1, 0, -1, -r_val]
    roots = np.roots(coeffs)
    real_roots = sorted([rt.real for rt in roots if abs(rt.imag) < 1e-8])
    if len(real_roots) == 3:
        return float(real_roots[1] - real_roots[0])
    return 0.0


def ecosystem_shift(T: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray, int]:
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
    return x, h, int(bif_t)


def rollout_eco(
    x0: float,
    h_start: float,
    h_end: float,
    N: int,
    H: int,
    eps: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
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


def eco_basin_width(h_val: float, s: float = 0.2) -> float:
    coeffs = [1, -1, s**2 + h_val, -s**2]
    roots = np.roots(coeffs)
    real_pos = sorted([rt.real for rt in roots if abs(rt.imag) < 1e-8 and rt.real > 0.01])
    if len(real_pos) >= 2:
        return float(real_pos[-1] - real_pos[-2])
    if len(real_pos) == 1:
        return float(real_pos[0])
    return 0.0


def compute_survival_timeseries(
    trajectory: np.ndarray,
    param_schedule: np.ndarray,
    rollout_fn: Callable,
    threshold: float,
    n_rollouts: int,
    horizon: int,
    perturb_std: float,
    step: int,
    survival_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(trajectory)
    times = np.arange(0, T, step)
    survival = np.zeros(len(times))

    for i, t in enumerate(times):
        x0 = trajectory[t]
        p_start = param_schedule[t]
        p_end = param_schedule[min(t + horizon, T - 1)]
        futures = rollout_fn(
            x0,
            p_start,
            p_end,
            n_rollouts,
            horizon,
            eps=perturb_std,
            seed=i,
        )
        endpoints = futures[:, -1]
        if survival_direction == "above":
            survival[i] = np.mean(endpoints > threshold)
        else:
            survival[i] = np.mean(endpoints < threshold)

    return times.astype(float), survival


def first_persistent_crossing(mask: np.ndarray, persistence: int) -> int | None:
    if persistence <= 1:
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) > 0 else None

    count = 0
    for i, flag in enumerate(mask):
        if flag:
            count += 1
            if count >= persistence:
                return i - persistence + 1
        else:
            count = 0
    return None


def compute_normalized_lead_time(
    survival: np.ndarray,
    times: np.ndarray,
    bif_time: int,
    total_T: int,
    k_zscore: float = 2.0,
    baseline_frac: float = 0.2,
    persistence: int = 3,
) -> tuple[float, float, float, float | None]:
    baseline_n = max(3, int(len(survival) * baseline_frac))
    base = survival[:baseline_n]

    mu0 = float(np.mean(base))
    sigma0 = float(np.std(base))
    threshold = mu0 - k_zscore * sigma0

    below = survival < threshold
    alert_idx = first_persistent_crossing(below, persistence=persistence)

    if alert_idx is None:
        return 0.0, threshold, mu0, None

    alert_time = float(times[alert_idx])

    # Roadmap operational definition: miss or late => 0.0
    if alert_time >= bif_time:
        return 0.0, threshold, mu0, alert_time

    lead = float((bif_time - alert_time) / max(total_T, 1))
    return lead, threshold, mu0, alert_time


def run_single_configuration(
    system: SystemSpec,
    seed: int,
    sigma: float,
    n_rollouts: int,
    horizon: int,
    step: int,
    min_width: float,
) -> dict:
    trajectory, p_schedule, bif_time = system.generator(system.n_steps, seed)
    times, survival = compute_survival_timeseries(
        trajectory=trajectory,
        param_schedule=p_schedule,
        rollout_fn=system.rollout,
        threshold=system.threshold,
        n_rollouts=n_rollouts,
        horizon=horizon,
        perturb_std=sigma,
        step=step,
        survival_direction=system.survival_direction,
    )

    basin_w = np.array([system.basin_width(p_schedule[int(t)]) for t in times])

    rho_sw = basin_contraction_correlation(survival, basin_w, min_width=min_width)
    mono_frac = monotonicity_fraction(survival, basin_width=basin_w, min_width=min_width)

    lead_norm, alert_threshold, baseline_mu, alert_time = compute_normalized_lead_time(
        survival=survival,
        times=times,
        bif_time=bif_time,
        total_T=len(trajectory),
    )

    valid = basin_w > min_width
    pre_collapse = times < bif_time
    valid_corr = valid & pre_collapse
    rho_tau = float("nan")
    if valid_corr.sum() >= 5:
        tau = bif_time - times[valid_corr]
        rho_tau = safe_spearman(survival[valid_corr], tau)

    return {
        "system": system.name,
        "seed": seed,
        "sigma": sigma,
        "rho_sw": rho_sw,
        "rho_tau": rho_tau,
        "monotonicity_frac": mono_frac,
        "lead_time_norm": lead_norm,
        "alert_time": alert_time,
        "bif_time": float(bif_time),
        "alert_threshold": alert_threshold,
        "baseline_mu": baseline_mu,
        "n_points": int(len(times)),
    }


def build_systems() -> list[SystemSpec]:
    return [
        SystemSpec(
            name="Saddle-Node",
            generator=saddle_node,
            n_steps=300,
            rollout=rollout_sn,
            basin_width=sn_basin_width,
            threshold=-0.5,
            survival_direction="above",
        ),
        SystemSpec(
            name="Double-Well",
            generator=double_well,
            n_steps=400,
            rollout=rollout_dw,
            basin_width=dw_basin_width,
            threshold=0.0,
            survival_direction="below",
        ),
        SystemSpec(
            name="Ecosystem",
            generator=ecosystem_shift,
            n_steps=400,
            rollout=rollout_eco,
            basin_width=eco_basin_width,
            threshold=0.3,
            survival_direction="above",
        ),
    ]


def summarize_table_1(df: pd.DataFrame, default_sigma: float) -> pd.DataFrame:
    table = (
        df[df["sigma"] == default_sigma]
        .groupby("system", as_index=False)
        .agg(
            rho_sw_mean=("rho_sw", "mean"),
            rho_sw_std=("rho_sw", "std"),
            mono_mean=("monotonicity_frac", "mean"),
            mono_std=("monotonicity_frac", "std"),
            lead_mean=("lead_time_norm", "mean"),
            lead_std=("lead_time_norm", "std"),
            alert_rate=("alert_time", lambda x: float(np.mean(pd.notna(x)))),
        )
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analytical S_t on synthetic systems")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.03, 0.05, 0.10],
        help="Perturbation std values",
    )
    parser.add_argument("--n-rollouts", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-width", type=float, default=0.05)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/synthetic",
        help="Output directory for csv files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = build_systems()

    records: list[dict] = []
    for sigma in args.sigmas:
        for system in systems:
            for seed in range(args.seeds):
                records.append(
                    run_single_configuration(
                        system=system,
                        seed=seed,
                        sigma=float(sigma),
                        n_rollouts=args.n_rollouts,
                        horizon=args.horizon,
                        step=args.step,
                        min_width=args.min_width,
                    )
                )

    run_df = pd.DataFrame.from_records(records)

    run_csv = out_dir / "run_level_metrics.csv"
    run_df.to_csv(run_csv, index=False)

    sigma_summary = (
        run_df.groupby(["system", "sigma"], as_index=False)
        .agg(
            rho_sw_mean=("rho_sw", "mean"),
            mono_mean=("monotonicity_frac", "mean"),
            lead_mean=("lead_time_norm", "mean"),
            alert_rate=("alert_time", lambda x: float(np.mean(pd.notna(x)))),
        )
        .sort_values(["system", "sigma"])
    )
    sigma_csv = out_dir / "sigma_sweep_summary.csv"
    sigma_summary.to_csv(sigma_csv, index=False)

    default_sigma = 0.05 if 0.05 in args.sigmas else float(args.sigmas[0])
    table_1 = summarize_table_1(run_df, default_sigma=default_sigma)
    table_csv = out_dir / "table1_summary.csv"
    table_1.to_csv(table_csv, index=False)

    print("=" * 72)
    print("Sprint 1 synthetic run complete")
    print("=" * 72)
    print(f"runs: {len(run_df)}")
    print(f"run-level csv: {run_csv}")
    print(f"sigma summary csv: {sigma_csv}")
    print(f"table 1 csv (sigma={default_sigma:.2f}): {table_csv}")
    print("\nTable 1 preview:")
    print(table_1.to_string(index=False))


if __name__ == "__main__":
    main()
