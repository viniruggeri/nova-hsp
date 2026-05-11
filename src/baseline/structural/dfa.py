"""Detrended Fluctuation Analysis (DFA) baseline.

Provides rolling alpha(t) as a structural early-warning indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baseline.types import BaselineResult


@dataclass
class DFABaseline:
    """Rolling DFA estimator.

    Higher alpha typically indicates stronger long-range dependence,
    often associated with critical slowing down.
    """

    window: int = 80
    min_scale: int = 4
    n_scales: int = 8
    step: int = 1

    def _dfa_alpha(self, segment: np.ndarray) -> float:
        n = len(segment)
        if n < max(self.window, self.min_scale * 2):
            return float("nan")

        x = segment - np.mean(segment)
        y = np.cumsum(x)

        max_scale = max(self.min_scale + 1, n // 4)
        if max_scale <= self.min_scale:
            return float("nan")

        scales = np.unique(
            np.logspace(
                np.log10(self.min_scale),
                np.log10(max_scale),
                self.n_scales,
            ).astype(int)
        )

        fluctuations = []
        used_scales = []
        for scale in scales:
            n_boxes = n // scale
            if n_boxes < 2:
                continue

            rms_vals = []
            for b in range(n_boxes):
                start = b * scale
                end = start + scale
                box = y[start:end]
                t = np.arange(scale)
                coeff = np.polyfit(t, box, 1)
                trend = np.polyval(coeff, t)
                rms_vals.append(np.sqrt(np.mean((box - trend) ** 2)))

            f_scale = float(np.sqrt(np.mean(np.square(rms_vals))))
            if f_scale > 0:
                fluctuations.append(f_scale)
                used_scales.append(scale)

        if len(used_scales) < 3:
            return float("nan")

        log_s = np.log(np.asarray(used_scales, dtype=float))
        log_f = np.log(np.asarray(fluctuations, dtype=float))
        slope, _ = np.polyfit(log_s, log_f, 1)
        return float(slope)

    def compute_indicator(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return rolling DFA alpha(t)."""
        if trajectory.ndim == 2:
            series = trajectory.mean(axis=1)
        else:
            series = trajectory

        if len(series) < self.window:
            return np.array([], dtype=float), np.array([], dtype=float)

        times = []
        values = []
        for end in range(self.window, len(series) + 1, self.step):
            seg = series[end - self.window : end]
            times.append(float(end - 1))
            values.append(self._dfa_alpha(seg))

        return np.asarray(times, dtype=float), np.asarray(values, dtype=float)

    @staticmethod
    def _first_persistent_crossing(mask: np.ndarray, persistence: int) -> int | None:
        count = 0
        for i, flag in enumerate(mask):
            if flag:
                count += 1
                if count >= persistence:
                    return i - persistence + 1
            else:
                count = 0
        return None

    def compute_baseline_result(
        self,
        trajectory: np.ndarray,
        persistence: int = 3,
        baseline_frac: float = 0.2,
        zscore_k: float = 2.0,
    ) -> BaselineResult:
        """Compute standardized baseline result for DFA alpha."""
        times, values = self.compute_indicator(trajectory)
        if len(values) == 0:
            return BaselineResult(
                name="DFA",
                times=times,
                values=values,
                alert_time=None,
                higher_means_risk=True,
            )

        baseline_n = max(3, int(len(values) * baseline_frac))
        base = values[:baseline_n]
        mu = float(np.nanmean(base))
        sigma = float(np.nanstd(base))
        threshold = mu + zscore_k * sigma

        mask = values > threshold
        alert_idx = self._first_persistent_crossing(mask, persistence=persistence)
        alert_time = float(times[alert_idx]) if alert_idx is not None else None

        return BaselineResult(
            name="DFA",
            times=times,
            values=values,
            alert_time=alert_time,
            higher_means_risk=True,
        )
