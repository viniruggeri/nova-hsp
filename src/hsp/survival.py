"""
Survival functions for learned HSP models.

Maps latent states back to survival decisions (alive/dead) for use
with compute_basin_access_learned().

Two strategies:
  1. DecoderSurvivalFn: uses a decoder z -> x_hat + threshold (semi-synthetic)
  2. LatentDistanceSurvivalFn: z within n-sigma of training distribution (real data)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderSurvivalFn:
    """
    Survival via decoder: map z -> x_hat, then apply threshold.

    For semi-synthetic validation where we know the observation-space
    threshold (e.g., x > -0.5 for saddle-node).

    Args:
        decoder: nn.Module mapping z (batch, latent_dim) -> x_hat (batch, obs_dim).
        threshold: Observation-space survival threshold.
        direction: 'above' means alive if x_hat > threshold.
    """

    def __init__(self, decoder: nn.Module, threshold: float, direction: str = "above"):
        self.decoder = decoder
        self.threshold = threshold
        self.direction = direction

    def __call__(self, z_batch: torch.Tensor) -> torch.Tensor:
        """z_batch: (N, latent_dim) -> (N,) of 0.0/1.0"""
        with torch.no_grad():
            x_hat = self.decoder(z_batch)  # (N, obs_dim)
            x_scalar = x_hat.squeeze(-1)   # (N,)
            if self.direction == "above":
                return (x_scalar > self.threshold).float()
            else:
                return (x_scalar < self.threshold).float()


class LatentDistanceSurvivalFn:
    """
    Survival via latent distance: alive if z is within n-sigma of the
    training latent distribution.

    For real data where no observation-space threshold is known.
    Operationalizes 'survival' as 'the system remains in the region
    it occupied during the stable pre-transition period'.

    Args:
        z_center: Mean of training latent states, shape (latent_dim,).
        z_scale: Std of training latent states, shape (latent_dim,).
        n_sigma: Number of standard deviations for the survival boundary.
    """

    def __init__(self, z_center: torch.Tensor, z_scale: torch.Tensor, n_sigma: float = 3.0):
        self.center = z_center
        self.scale = z_scale.clamp(min=1e-6)  # avoid division by zero
        self.n_sigma = n_sigma

    def __call__(self, z_batch: torch.Tensor) -> torch.Tensor:
        """z_batch: (N, latent_dim) -> (N,) of 0.0/1.0"""
        # Standardized distance per dimension, then L2 norm
        standardized = (z_batch - self.center) / self.scale
        dist = standardized.norm(dim=-1)  # (N,)
        return (dist < self.n_sigma).float()
