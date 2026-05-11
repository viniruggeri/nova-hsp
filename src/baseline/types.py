"""Shared baseline result types for geometric early warning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    """Standardized output for baseline indicators.

    Attributes:
        name: Baseline name.
        times: Indicator timestamps.
        values: Indicator values at each timestamp.
        alert_time: First alert timestamp, if any.
        higher_means_risk: Whether larger values indicate higher risk.
    """

    name: str
    times: np.ndarray
    values: np.ndarray
    alert_time: float | None
    higher_means_risk: bool
