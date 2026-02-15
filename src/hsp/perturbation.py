"""
Perturbation Module — generates ε samples for basin access estimation.

Architecture v3 pipeline:
    z_t → [Perturbation] → {z_t + ε_1, ..., z_t + ε_N}

Supports different perturbation strategies:
    - GaussianPerturbation: ε ~ N(0, σ²I) — standard, used in all NB 09-11
    - UniformBallPerturbation: ε ~ Uniform(B(0, r)) — bounded support
    - TruncatedGaussianPerturbation: ε ~ N(0, σ²I) truncated at ||ε|| ≤ r

The bounded support variants satisfy Assumption A2 (bounded perturbations)
of Proposition 1 exactly, rather than approximately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePerturbation(ABC):
    """Base class for perturbation strategies."""

    @abstractmethod
    def sample(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Generate perturbed copies of z.

        Args:
            z: State to perturb, shape (d,) or (batch, d).
            n_samples: Number of perturbation samples N.

        Returns:
            Perturbed states, shape (n_samples, d) or (batch, n_samples, d).
        """
        ...

    @abstractmethod
    def bound(self) -> float:
        """Return the perturbation bound ε̄ (sup of ||ε||)."""
        ...


class GaussianPerturbation(BasePerturbation):
    """
    Standard Gaussian perturbation: ε ~ N(0, σ²I).

    This is the default used throughout NB 09-11.
    Note: technically unbounded, but P(||ε|| > 3σ) < 0.3%.

    Args:
        std: Standard deviation σ.
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def sample(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        if z.dim() == 1:
            eps = torch.randn(n_samples, z.shape[0], device=z.device) * self.std
            return z.unsqueeze(0) + eps  # (N, d)
        else:
            B, d = z.shape
            eps = torch.randn(B, n_samples, d, device=z.device) * self.std
            return z.unsqueeze(1) + eps  # (B, N, d)

    def bound(self) -> float:
        return 3.0 * self.std  # 99.7% bound


class UniformBallPerturbation(BasePerturbation):
    """
    Uniform perturbation in L2 ball: ε ~ Uniform(B(0, r)).

    Satisfies A2 (bounded perturbations) exactly.

    Args:
        radius: Ball radius r = ε̄.
    """

    def __init__(self, radius: float = 0.1):
        self.radius = radius

    def sample(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        d = z.shape[-1]
        # Sample uniformly in d-ball via normalized Gaussian * uniform radius
        raw = torch.randn(*z.shape[:-1], n_samples, d, device=z.device)
        raw = raw / raw.norm(dim=-1, keepdim=True)  # unit sphere
        r = torch.rand(*z.shape[:-1], n_samples, 1, device=z.device)
        r = r.pow(1.0 / d) * self.radius  # uniform in volume
        eps = raw * r

        if z.dim() == 1:
            return z.unsqueeze(0) + eps  # (N, d)
        else:
            return z.unsqueeze(-2) + eps  # (B, N, d)

    def bound(self) -> float:
        return self.radius
