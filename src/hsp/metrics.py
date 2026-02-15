"""
HSP v3 Metrics — geometric early warning evaluation.

These metrics evaluate S_t as a basin access estimator, NOT as a
TTF regressor. Replaces RMSE/C-index/NASA Score with geometry-aware metrics.

Metrics:
    - basin_contraction_correlation: ρ(S_t, W_t) — does S_t track basin width?
    - lead_time_to_basin_collapse: how early does S_t alert?
    - separability_score: AUC for pre- vs post-basin-loss classification
    - monotonicity_fraction: fraction of steps where S_{t+1} ≤ S_t
    - partial_correlation: ρ(S_t, τ | EWS) — info beyond classical EWS

References:
    - NB 10.1 (structural validation): Tests 1-3
    - NB 11 (monotonicity): Verifications 1-3
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def basin_contraction_correlation(
    survival: np.ndarray,
    basin_width: np.ndarray,
    min_width: float = 0.05,
) -> float:
    """
    Spearman correlation between S_t and theoretical basin width W_t.

    Filters out post-collapse region where W < min_width.

    Args:
        survival: S_t series.
        basin_width: W_t series (same length as survival).
        min_width: Minimum basin width to include.

    Returns:
        Spearman ρ(S_t, W_t). Expected: > +0.8 for well-behaved systems.
    """
    valid = basin_width > min_width
    if valid.sum() < 5:
        return float("nan")

    rho, _ = spearmanr(survival[valid], basin_width[valid])
    return float(rho)


def monotonicity_fraction(
    survival: np.ndarray,
    basin_width: np.ndarray | None = None,
    min_width: float = 0.05,
) -> float:
    """
    Fraction of consecutive steps where S_{t+1} ≤ S_t (pre-collapse).

    Under Proposition 1, this should approach 1.0 as N_rollouts → ∞,
    when all 5 assumptions hold.

    Args:
        survival: S_t series.
        basin_width: If provided, filters to pre-collapse region.
        min_width: Minimum basin width for filtering.

    Returns:
        Fraction in [0, 1]. Expected: > 0.80 for SN/ECO, ~0.54 for DW.
    """
    if basin_width is not None:
        valid = basin_width > min_width
        s = survival[valid]
    else:
        s = survival

    if len(s) < 2:
        return float("nan")

    ds = np.diff(s)
    return float(np.mean(ds <= 0))


def violation_anatomy(
    survival: np.ndarray,
    times: np.ndarray,
    bif_time: float,
    basin_width: np.ndarray | None = None,
    min_width: float = 0.05,
) -> dict:
    """
    Analyze monotonicity violations: where do they occur, how large are they?

    Args:
        survival: S_t series.
        times: Timestamps corresponding to S_t.
        bif_time: Known bifurcation time.
        basin_width: Optional basin width for filtering.
        min_width: Minimum basin width.

    Returns:
        Dictionary with:
            - n_violations: count
            - violation_mean: mean magnitude of ΔS > 0
            - violation_max: max magnitude
            - median_tau_violations: median distance-to-bifurcation of violations
            - median_tau_monotone: median distance-to-bifurcation of monotone steps
            - rho_tau_magnitude: ρ(τ, |violation|) — expected negative
    """
    if basin_width is not None:
        valid = basin_width > min_width
        s = survival[valid]
        t = times[valid]
    else:
        s = survival
        t = times

    ds = np.diff(s)
    tau = bif_time - t[1:]  # distance to bifurcation for each ΔS

    viol_mask = ds > 0
    violations = ds[viol_mask]
    tau_viol = tau[viol_mask]
    tau_ok = tau[~viol_mask]

    result: dict = {
        "n_violations": int(viol_mask.sum()),
        "n_monotone": int((~viol_mask).sum()),
        "violation_mean": float(violations.mean()) if len(violations) > 0 else 0.0,
        "violation_max": float(violations.max()) if len(violations) > 0 else 0.0,
        "median_tau_violations": float(np.median(tau_viol)) if len(tau_viol) > 0 else float("nan"),
        "median_tau_monotone": float(np.median(tau_ok)) if len(tau_ok) > 0 else float("nan"),
    }

    # Correlation: violations larger near bifurcation?
    if len(tau_viol) > 3:
        rho, _ = spearmanr(tau_viol, violations)
        result["rho_tau_magnitude"] = float(rho)
    else:
        result["rho_tau_magnitude"] = float("nan")

    return result


def separability_score(
    survival: np.ndarray,
    basin_width: np.ndarray,
    width_threshold: float = 0.1,
) -> float:
    """
    AUC for classifying states as pre- vs post-basin-loss using S_t.

    Labels: basin_width > width_threshold → "pre" (positive class).
    Score: S_t as discriminant.

    Args:
        survival: S_t series.
        basin_width: W_t series.
        width_threshold: Threshold for "basin still exists".

    Returns:
        AUC-ROC in [0, 1]. Expected: > 0.85 for well-behaved systems.
    """
    from sklearn.metrics import roc_auc_score

    labels = (basin_width > width_threshold).astype(int)

    # Need both classes present
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")

    return float(roc_auc_score(labels, survival))


def partial_correlation_vs_ews(
    survival: np.ndarray,
    tau: np.ndarray,
    ews_dict: dict[str, np.ndarray],
) -> float:
    """
    Partial Spearman correlation ρ(S_t, τ | EWS).

    Measures information in S_t about proximity to bifurcation
    beyond what's captured by classical early warning signals.

    Uses rank-based regression residuals for partial correlation.

    Args:
        survival: S_t series.
        tau: Distance to bifurcation τ = t_bif - t.
        ews_dict: Dictionary of EWS series {"variance": [...], "ac1": [...], ...}.

    Returns:
        Partial Spearman ρ. Expected: > +0.1 (S_t adds info beyond EWS).
    """
    from scipy.stats import rankdata

    n = len(survival)
    if n < 10:
        return float("nan")

    # Rank everything
    rank_s = rankdata(survival)
    rank_tau = rankdata(tau)

    # Build EWS matrix
    ews_names = sorted(ews_dict.keys())
    if len(ews_names) == 0:
        rho, _ = spearmanr(survival, tau)
        return float(rho)

    ews_matrix = np.column_stack([rankdata(ews_dict[k]) for k in ews_names])

    # Residualize S and τ on EWS (OLS on ranks)
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
