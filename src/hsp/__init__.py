"""
HSP v3 — Basin Access Probability Framework.

Core idea: S_t = P[Φ^H(x_t + ε) ∈ B(p_{t+H})]
estimates basin accessibility as a geometric early warning signal
for critical transitions.

Modules:
    basin_access: Core S_t computation (analytical + learned dynamics)
    encoder: Raw observations → latent state z_t
    dynamics: Learned dynamics g_θ for rollouts in latent space
    perturbation: Perturbation sampling (Gaussian, uniform ball)
    detector: Collapse detection via K-consecutive rule
    metrics: Geometric evaluation metrics (replaces RMSE/C-index)
"""

# --- Core S_t computation ---
from src.hsp.basin_access import (
    BasinAccessConfig,
    BasinAccessResult,
    compute_basin_access,
    compute_basin_access_learned,
)

# --- Encoder ---
from src.hsp.encoder import (
    IdentityEncoder,
    LSTMEncoder,
    TransformerEncoder,
)

# --- Dynamics models ---
from src.hsp.dynamics import (
    IdentityDynamics,
    LatentDynamicsMLP,
    ResidualDynamics,
    NeuralODE,
)

# --- Perturbation ---
from src.hsp.perturbation import (
    BasePerturbation,
    GaussianPerturbation,
    UniformBallPerturbation,
)

# --- Detector ---
from src.hsp.detector import (
    DetectorConfig,
    DetectionResult,
    detect_collapse,
)

# --- Metrics ---
from src.hsp.metrics import (
    basin_contraction_correlation,
    monotonicity_fraction,
    violation_anatomy,
    separability_score,
    partial_correlation_vs_ews,
)

__all__ = [
    # basin_access
    "BasinAccessConfig",
    "BasinAccessResult",
    "compute_basin_access",
    "compute_basin_access_learned",
    # encoder
    "IdentityEncoder",
    "LSTMEncoder",
    "TransformerEncoder",
    # dynamics
    "IdentityDynamics",
    "LatentDynamicsMLP",
    "ResidualDynamics",
    "NeuralODE",
    # perturbation
    "BasePerturbation",
    "GaussianPerturbation",
    "UniformBallPerturbation",
    # detector
    "DetectorConfig",
    "DetectionResult",
    "detect_collapse",
    # metrics
    "basin_contraction_correlation",
    "monotonicity_fraction",
    "violation_anatomy",
    "separability_score",
    "partial_correlation_vs_ews",
]
