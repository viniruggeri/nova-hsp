"""
Collapse Detector — triggers alert when S_t indicates basin loss.

Architecture v3 pipeline:
    S_t → [Detector] → alert (bool) + alert_time

Detection rule (unsupervised):
    Alert when S_t < δ for K consecutive steps.

This is intentionally simple — the intelligence is in S_t, not the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorConfig:
    """Configuration for collapse detection."""

    threshold: float = 0.5
    """S_t threshold δ below which system is considered at risk."""

    persistence: int = 3
    """Number of consecutive steps K below threshold to trigger alert."""


@dataclass
class DetectionResult:
    """Result of collapse detection."""

    alert: bool
    """Whether a collapse alert was triggered."""

    alert_time: float | None
    """Time of first alert (None if no alert)."""

    collapse_time: float | None
    """Time of actual collapse (None if unknown/not provided)."""

    lead_time: float | None
    """collapse_time - alert_time (None if either is missing)."""

    below_threshold: np.ndarray
    """Boolean mask: True where S_t < δ."""


def detect_collapse(
    times: np.ndarray,
    survival: np.ndarray,
    config: DetectorConfig | None = None,
    collapse_time: float | None = None,
) -> DetectionResult:
    """
    Detect collapse from basin access probability series.

    Args:
        times: Timestamps of S_t measurements.
        survival: S_t values at each timestamp.
        config: Detection configuration.
        collapse_time: Known collapse time (for lead time computation).

    Returns:
        DetectionResult with alert status and timing.
    """
    if config is None:
        config = DetectorConfig()

    below = survival < config.threshold

    # Find first run of K consecutive below-threshold
    alert_time = None
    if config.persistence <= 1:
        # Any single crossing triggers
        idx = np.where(below)[0]
        if len(idx) > 0:
            alert_time = times[idx[0]]
    else:
        # Need K consecutive
        count = 0
        for i, b in enumerate(below):
            if b:
                count += 1
                if count >= config.persistence:
                    # Alert at the START of the run
                    alert_time = times[i - config.persistence + 1]
                    break
            else:
                count = 0

    lead_time = None
    if alert_time is not None and collapse_time is not None:
        lead_time = collapse_time - alert_time

    return DetectionResult(
        alert=alert_time is not None,
        alert_time=alert_time,
        collapse_time=collapse_time,
        lead_time=lead_time,
        below_threshold=below,
    )
