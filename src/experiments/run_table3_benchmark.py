"""Run unified synthetic benchmark for Table 3.

Compares analytical S_t against:
- B1 Rolling Variance
- B2 Rolling AC1
- B3 Rolling Skewness
- B4 DFA
- B5 Basin Stability
- Cox PH hazard baseline

Outputs:
- benchmark_run_level.csv
- benchmark_summary.csv
- table3_main_comparison.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from src.baseline.heuristics.early_warning import EarlyWarningSignals
from src.baseline.structural import BasinStabilityBaseline, DFABaseline
from src.baseline.survival.cox_ph import CoxPHModel
from src.evaluation.aggregator import ResultsAggregator
from src.experiments.run_hsp_synthetic import (
    SystemSpec,
    build_systems,
    compute_normalized_lead_time,
    compute_survival_timeseries,
)


@dataclass
class RunCase:
    system_name: str
    system: SystemSpec
    seed: int
    sigma: float
    trajectory: np.ndarray
    p_schedule: np.ndarray
    bif_time: int

    @property
    def run_id(self) -> str:
        return f"{self.system_name}|seed={self.seed}|sigma={self.sigma:.3f}"

    @property
    def total_T(self) -> int:
        return int(len(self.trajectory))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho = spearmanr(x, y)[0]
    return float(np.asarray(rho).item())


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


def stability_signal(values: np.ndarray, higher_means_risk: bool) -> np.ndarray:
    return -values if higher_means_risk else values


def compute_alert_time(
    times: np.ndarray,
    values: np.ndarray,
    higher_means_risk: bool,
    baseline_frac: float,
    zscore_k: float,
    persistence: int,
) -> float | None:
    if len(values) < 3:
        return None

    baseline_n = max(3, int(len(values) * baseline_frac))
    base = values[:baseline_n]
    mu = float(np.mean(base))
    sigma = float(np.std(base))

    if higher_means_risk:
        threshold = mu + zscore_k * sigma
        mask = values > threshold
    else:
        threshold = mu - zscore_k * sigma
        mask = values < threshold

    idx = first_persistent_crossing(mask, persistence=persistence)
    return float(times[idx]) if idx is not None else None


def normalized_lead_time_from_alert(
    alert_time: float | None,
    bif_time: int,
    total_T: int,
) -> float:
    if alert_time is None or alert_time >= bif_time:
        return 0.0
    return float((bif_time - alert_time) / max(total_T, 1))


def rho_tau_metric(
    times: np.ndarray,
    values: np.ndarray,
    bif_time: int,
    higher_means_risk: bool,
) -> float:
    mask = times < bif_time
    if int(mask.sum()) < 5:
        return float("nan")

    tau = bif_time - times[mask]
    s = stability_signal(values[mask], higher_means_risk=higher_means_risk)
    if np.std(s) < 1e-12 or np.std(tau) < 1e-12:
        return float("nan")
    return safe_spearman(s, tau)


def separability_metric(
    values: np.ndarray,
    basin_width: np.ndarray,
    higher_means_risk: bool,
    width_threshold: float = 0.1,
) -> float:
    labels = (basin_width > width_threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")

    scores = stability_signal(values, higher_means_risk=higher_means_risk)
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def basin_width_series(system: SystemSpec, p_schedule: np.ndarray, times: np.ndarray) -> np.ndarray:
    idx = np.clip(times.astype(int), 0, len(p_schedule) - 1)
    return np.asarray([system.basin_width(float(p_schedule[i])) for i in idx], dtype=float)


def make_basin_rollout(
    system_name: str,
    sigma: float,
    seed: int,
) -> Callable[[np.ndarray, float, int], np.ndarray]:
    rng = np.random.RandomState(seed)

    def rollout_sn(x0_batch: np.ndarray, p_t: float, horizon: int) -> np.ndarray:
        dt, noise = 0.02, 0.08
        x = np.asarray(x0_batch, dtype=float).reshape(-1)
        for _ in range(horizon):
            x = x + dt * (p_t - x**2) + rng.normal(0, noise + sigma * 0.1, size=x.shape)
            x = np.clip(x, -5.0, 5.0)
        return x

    def rollout_dw(x0_batch: np.ndarray, p_t: float, horizon: int) -> np.ndarray:
        dt, noise = 0.01, 0.12
        x = np.asarray(x0_batch, dtype=float).reshape(-1)
        for _ in range(horizon):
            x = x + dt * (x - x**3 + p_t) + rng.normal(0, noise + sigma * 0.1, size=x.shape)
            x = np.clip(x, -3.0, 3.0)
        return x

    def rollout_eco(x0_batch: np.ndarray, p_t: float, horizon: int) -> np.ndarray:
        dt, noise, r_g, K, s = 0.05, 0.02, 1.0, 1.0, 0.2
        x = np.asarray(x0_batch, dtype=float).reshape(-1)
        x = np.clip(x, 0.01, 3.0)
        for _ in range(horizon):
            grazing = p_t * x**2 / (x**2 + s**2)
            growth = r_g * x * (1.0 - x / K)
            x = x + dt * (growth - grazing) + rng.normal(0, noise + sigma * 0.1, size=x.shape)
            x = np.clip(x, 0.01, 3.0)
        return x

    if system_name == "Saddle-Node":
        return rollout_sn
    if system_name == "Double-Well":
        return rollout_dw
    if system_name == "Ecosystem":
        return rollout_eco
    raise ValueError(f"Unknown system for basin rollout: {system_name}")


def evaluate_model_series(
    model_name: str,
    case: RunCase,
    times: np.ndarray,
    values: np.ndarray,
    higher_means_risk: bool,
    min_width: float,
    baseline_frac: float,
    zscore_k: float,
    persistence: int,
) -> dict:
    widths = basin_width_series(case.system, case.p_schedule, times)
    valid_rho = widths > min_width

    alert_time = compute_alert_time(
        times=times,
        values=values,
        higher_means_risk=higher_means_risk,
        baseline_frac=baseline_frac,
        zscore_k=zscore_k,
        persistence=persistence,
    )
    lead = normalized_lead_time_from_alert(alert_time, case.bif_time, case.total_T)

    rho_tau = float("nan")
    if int(valid_rho.sum()) >= 5:
        rho_tau = rho_tau_metric(
            times=times[valid_rho],
            values=values[valid_rho],
            bif_time=case.bif_time,
            higher_means_risk=higher_means_risk,
        )

    sep = separability_metric(
        values=values,
        basin_width=widths,
        higher_means_risk=higher_means_risk,
    )

    return {
        "run_id": case.run_id,
        "system": case.system_name,
        "seed": case.seed,
        "sigma": case.sigma,
        "model": model_name,
        "rho_tau": rho_tau,
        "lead_time_norm": lead,
        "separability": sep,
        "alert_time": alert_time,
        "bif_time": float(case.bif_time),
        "n_points": int(len(times)),
    }


def build_cases(systems: Sequence[SystemSpec], seeds: int, sigmas: list[float]) -> list[RunCase]:
    cases: list[RunCase] = []
    for sigma in sigmas:
        for system in systems:
            for seed in range(seeds):
                trajectory, p_schedule, bif_time = system.generator(system.n_steps, seed)
                cases.append(
                    RunCase(
                        system_name=system.name,
                        system=system,
                        seed=seed,
                        sigma=float(sigma),
                        trajectory=trajectory,
                        p_schedule=p_schedule,
                        bif_time=int(bif_time),
                    )
                )
    return cases


def fit_cox(cases: list[RunCase], penalizer: float) -> CoxPHModel | None:
    X = []
    durations = []
    for case in cases:
        series = case.trajectory if case.trajectory.ndim == 1 else case.trajectory.mean(axis=1)
        X.append(
            [
                float(series[-1]),
                float(np.mean(series)),
                float(np.std(series)),
                float(np.max(series)),
                float(len(series)),
            ]
        )
        durations.append(float(case.bif_time))

    model = CoxPHModel(penalizer=penalizer, normalize=True)
    try:
        model.fit(np.asarray(X, dtype=float), np.asarray(durations, dtype=float), np.ones(len(cases), dtype=int))
        return model
    except Exception:
        return None


def cox_series_for_case(
    model: CoxPHModel,
    case: RunCase,
    times: np.ndarray,
) -> np.ndarray:
    series = case.trajectory if case.trajectory.ndim == 1 else case.trajectory.mean(axis=1)
    vals = []
    for t in times.astype(int):
        prefix = series[: max(1, t + 1)]
        feats = np.asarray(
            [
                [
                    float(prefix[-1]),
                    float(np.mean(prefix)),
                    float(np.std(prefix)),
                    float(np.max(prefix)),
                    float(len(prefix)),
                ]
            ],
            dtype=float,
        )
        vals.append(float(model.predict_risk(feats)[0]))
    return np.asarray(vals, dtype=float)


def pairwise_vs_st(
    run_df: pd.DataFrame,
    baseline_names: list[str],
) -> pd.DataFrame:
    stats_rows = []
    st = run_df[run_df["model"] == "S_t"].sort_values("run_id")

    for name in baseline_names:
        base = run_df[run_df["model"] == name].sort_values("run_id")
        merged = st[["run_id", "lead_time_norm", "separability"]].merge(
            base[["run_id", "lead_time_norm", "separability"]],
            on="run_id",
            suffixes=("_st", "_base"),
        )

        lead_x = merged["lead_time_norm_st"].to_numpy(dtype=float)
        lead_y = merged["lead_time_norm_base"].to_numpy(dtype=float)
        sep_x = merged["separability_st"].to_numpy(dtype=float)
        sep_y = merged["separability_base"].to_numpy(dtype=float)

        lead_mask = np.isfinite(lead_x) & np.isfinite(lead_y)
        sep_mask = np.isfinite(sep_x) & np.isfinite(sep_y)
        lead_xf, lead_yf = lead_x[lead_mask], lead_y[lead_mask]
        sep_xf, sep_yf = sep_x[sep_mask], sep_y[sep_mask]

        lead_p = float("nan")
        sep_p = float("nan")

        if len(lead_xf) >= 2:
            try:
                _, lead_p = wilcoxon(lead_xf, lead_yf)
            except ValueError:
                pass
        if len(sep_xf) >= 2:
            try:
                _, sep_p = wilcoxon(sep_xf, sep_yf)
            except ValueError:
                pass

        lead_delta = ResultsAggregator.cliffs_delta(lead_xf, lead_yf) if len(lead_xf) > 0 else float("nan")
        sep_delta = ResultsAggregator.cliffs_delta(sep_xf, sep_yf) if len(sep_xf) > 0 else float("nan")

        stats_rows.append(
            {
                "model": name,
                "wilcoxon_p_lead": lead_p,
                "cliffs_delta_lead": lead_delta,
                "cliffs_mag_lead": ResultsAggregator.cliffs_delta_magnitude(lead_delta),
                "wilcoxon_p_sep": sep_p,
                "cliffs_delta_sep": sep_delta,
                "cliffs_mag_sep": ResultsAggregator.cliffs_delta_magnitude(sep_delta),
            }
        )

    return pd.DataFrame(stats_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 3 unified benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    parser.add_argument("--n-rollouts", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-width", type=float, default=0.05)
    parser.add_argument("--zscore-k", type=float, default=2.0)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--baseline-frac", type=float, default=0.2)
    parser.add_argument("--bs-samples", type=int, default=150)
    parser.add_argument("--bs-radius", type=float, default=0.10)
    parser.add_argument("--cox-penalizer", type=float, default=1.0)
    parser.add_argument("--skip-cox", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = build_systems()
    cases = build_cases(systems=systems, seeds=args.seeds, sigmas=list(args.sigmas))

    run_rows: list[dict] = []
    st_times_cache: dict[str, np.ndarray] = {}

    ews = EarlyWarningSignals(window=20)
    dfa = DFABaseline(window=80, step=args.step)
    bs = BasinStabilityBaseline(
        n_samples=args.bs_samples,
        radius=args.bs_radius,
        horizon=args.horizon,
        step=args.step,
    )

    for case in cases:
        times_st, s_t = compute_survival_timeseries(
            trajectory=case.trajectory,
            param_schedule=case.p_schedule,
            rollout_fn=case.system.rollout,
            threshold=case.system.threshold,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            perturb_std=case.sigma,
            step=args.step,
            survival_direction=case.system.survival_direction,
        )
        st_times_cache[case.run_id] = times_st

        # S_t uses roadmap-defined alert rule.
        lead_st, _, _, alert_st = compute_normalized_lead_time(
            survival=s_t,
            times=times_st,
            bif_time=case.bif_time,
            total_T=case.total_T,
            k_zscore=args.zscore_k,
            baseline_frac=args.baseline_frac,
            persistence=args.persistence,
        )
        widths_st = basin_width_series(case.system, case.p_schedule, times_st)
        valid_st = widths_st > args.min_width
        rho_st = float("nan")
        if int(valid_st.sum()) >= 5:
            rho_st = rho_tau_metric(
                times=times_st[valid_st],
                values=s_t[valid_st],
                bif_time=case.bif_time,
                higher_means_risk=False,
            )
        sep_st = separability_metric(
            values=s_t,
            basin_width=widths_st,
            higher_means_risk=False,
        )
        run_rows.append(
            {
                "run_id": case.run_id,
                "system": case.system_name,
                "seed": case.seed,
                "sigma": case.sigma,
                "model": "S_t",
                "rho_tau": rho_st,
                "lead_time_norm": float(lead_st),
                "separability": sep_st,
                "alert_time": alert_st,
                "bif_time": float(case.bif_time),
                "n_points": int(len(times_st)),
            }
        )

        # B1-B3
        for name, indicator, higher in [
            ("B1 Rolling Variance", "variance", True),
            ("B2 Rolling AC1", "ac1", True),
            ("B3 Rolling Skewness", "skewness", False),
        ]:
            t, v = ews.compute_indicator(case.trajectory, indicator=indicator, step=args.step)
            run_rows.append(
                evaluate_model_series(
                    model_name=name,
                    case=case,
                    times=t,
                    values=v,
                    higher_means_risk=higher,
                    min_width=args.min_width,
                    baseline_frac=args.baseline_frac,
                    zscore_k=args.zscore_k,
                    persistence=args.persistence,
                )
            )

        # B4
        t_dfa, v_dfa = dfa.compute_indicator(case.trajectory)
        run_rows.append(
            evaluate_model_series(
                model_name="B4 DFA",
                case=case,
                times=t_dfa,
                values=np.nan_to_num(v_dfa, nan=0.0),
                higher_means_risk=True,
                min_width=args.min_width,
                baseline_frac=args.baseline_frac,
                zscore_k=args.zscore_k,
                persistence=args.persistence,
            )
        )

        # B5
        rollout_fn = make_basin_rollout(case.system_name, case.sigma, seed=case.seed + 100)
        if case.system.survival_direction == "above":
            def target_fn(x_final: np.ndarray, thr: float = case.system.threshold) -> np.ndarray:
                return np.asarray(x_final) > thr
        else:
            def target_fn(x_final: np.ndarray, thr: float = case.system.threshold) -> np.ndarray:
                return np.asarray(x_final) < thr

        t_bs, v_bs = bs.compute_indicator(
            trajectory=case.trajectory,
            param_schedule=case.p_schedule,
            rollout_fn=rollout_fn,
            target_fn=target_fn,
            seed=case.seed + 200,
        )
        run_rows.append(
            evaluate_model_series(
                model_name="B5 Basin Stability",
                case=case,
                times=t_bs,
                values=v_bs,
                higher_means_risk=False,
                min_width=args.min_width,
                baseline_frac=args.baseline_frac,
                zscore_k=args.zscore_k,
                persistence=args.persistence,
            )
        )

    # Cox PH baseline after all runs are available.
    if not args.skip_cox:
        cox_model = fit_cox(cases=cases, penalizer=args.cox_penalizer)
        if cox_model is not None:
            for case in cases:
                t = st_times_cache[case.run_id]
                vals = cox_series_for_case(cox_model, case, t)
                run_rows.append(
                    evaluate_model_series(
                        model_name="Cox PH (hazard)",
                        case=case,
                        times=t,
                        values=vals,
                        higher_means_risk=True,
                        min_width=args.min_width,
                        baseline_frac=args.baseline_frac,
                        zscore_k=args.zscore_k,
                        persistence=args.persistence,
                    )
                )

    run_df = pd.DataFrame(run_rows)

    summary = (
        run_df.groupby("model", as_index=False)
        .agg(
            rho_tau_mean=("rho_tau", "mean"),
            rho_tau_std=("rho_tau", "std"),
            lead_mean=("lead_time_norm", "mean"),
            lead_std=("lead_time_norm", "std"),
            separability_mean=("separability", "mean"),
            separability_std=("separability", "std"),
            alert_rate=("alert_time", lambda x: float(np.mean(pd.notna(x)))),
        )
        .sort_values("model")
    )

    baseline_names = [m for m in summary["model"].tolist() if m != "S_t"]
    stats_df = pairwise_vs_st(run_df=run_df, baseline_names=baseline_names)
    table3 = summary.merge(stats_df, on="model", how="left")

    run_csv = out_dir / "benchmark_run_level.csv"
    summary_csv = out_dir / "benchmark_summary.csv"
    table3_csv = out_dir / "table3_main_comparison.csv"

    run_df.to_csv(run_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    table3.to_csv(table3_csv, index=False)

    print("=" * 72)
    print("Table 3 synthetic benchmark complete")
    print("=" * 72)
    print(f"runs x models rows: {len(run_df)}")
    print(f"run-level csv: {run_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"table3 csv: {table3_csv}")
    print("\nTable 3 preview:")
    print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
