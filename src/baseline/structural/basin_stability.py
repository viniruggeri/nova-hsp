"""Basin Stability baseline (Menck et al. style Monte Carlo estimator).

This implementation computes a time-indexed stability proxy by sampling
initial conditions around each operating point and estimating the fraction
that converges to a target set under the provided dynamics rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.baseline.types import BaselineResult


@dataclass
class BasinStabilityBaseline:
    """Monte Carlo basin stability indicator.

    Args:
        n_samples: Number of sampled initial conditions per time point.
        radius: Sampling radius around nominal state.
        horizon: Rollout horizon to evaluate convergence.
        step: Temporal step for indicator time series.
        sampler: Sampling function (state, radius, n_samples, rng) -> samples.
            If None, isotropic Gaussian sampling is used.
    """

    n_samples: int = 300
    radius: float = 0.1
    horizon: int = 80
    step: int = 5
    sampler: Callable[[np.ndarray, float, int, np.random.RandomState], np.ndarray] | None = None

    def _default_sampler(
        self,
        state: np.ndarray,
        radius: float,
        n_samples: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        noise = rng.normal(0.0, radius, size=(n_samples,) + state.shape)
        return state[None, ...] + noise

    def compute_indicator(
        self,
        trajectory: np.ndarray,
        param_schedule: np.ndarray,
        rollout_fn: Callable[[np.ndarray, float, int], np.ndarray],
        target_fn: Callable[[np.ndarray], np.ndarray],
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute basin stability BS_t over time.

        Args:
            trajectory: Nominal state trajectory (T,) or (T, d).
            param_schedule: Parameter schedule p(t), shape (T,).
            rollout_fn: Dynamics rollout (x0_batch, p_t, horizon) -> x_final_batch.
            target_fn: Returns boolean mask for membership in target basin.
            seed: Random seed for Monte Carlo sampling.
        """
        if trajectory.ndim == 1:
            states = trajectory[:, None]
        else:
            states = trajectory

        rng = np.random.RandomState(seed)
        sample = self.sampler if self.sampler is not None else self._default_sampler

        times = np.arange(0, len(states), self.step, dtype=float)
        values = np.zeros(len(times), dtype=float)

        for i, t in enumerate(times.astype(int)):
            x_t = states[t]
            p_t = float(param_schedule[t])
            x0 = sample(x_t, self.radius, self.n_samples, rng)
            x_final = rollout_fn(x0, p_t, self.horizon)
            in_basin = target_fn(x_final)
            values[i] = float(np.mean(in_basin))

        return times, values

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
        param_schedule: np.ndarray,
        rollout_fn: Callable[[np.ndarray, float, int], np.ndarray],
        target_fn: Callable[[np.ndarray], np.ndarray],
        persistence: int = 3,
        baseline_frac: float = 0.2,
        zscore_k: float = 2.0,
        seed: int = 42,
    ) -> BaselineResult:
        """Compute standardized baseline result for basin stability."""
        times, values = self.compute_indicator(
            trajectory=trajectory,
            param_schedule=param_schedule,
            rollout_fn=rollout_fn,
            target_fn=target_fn,
            seed=seed,
        )

        baseline_n = max(3, int(len(values) * baseline_frac))
        base = values[:baseline_n]
        mu = float(np.mean(base))
        sigma = float(np.std(base))
        threshold = mu - zscore_k * sigma

        mask = values < threshold
        alert_idx = self._first_persistent_crossing(mask, persistence=persistence)
        alert_time = float(times[alert_idx]) if alert_idx is not None else None

        return BaselineResult(
            name="Basin Stability",
            times=times,
            values=values,
            alert_time=alert_time,
            higher_means_risk=False,
        )
