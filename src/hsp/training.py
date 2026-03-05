"""
Training infrastructure for learned HSP models.

Provides:
  - TimeSeriesWindowDataset: sliding-window dataset for encoder training
  - HSPLearnedPipeline: encoder + dynamics + decoder as a single module
  - HSPTrainer: training loop with multi-step prediction loss

Architecture:
    x_{t-W:t} -> [Encoder] -> z_t -> [Dynamics]^K -> z_{t+1..t+K}
                                   -> [Decoder] -> x_hat_t

Loss = multi-step prediction + reconstruction auxiliary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.hsp.encoder import LSTMEncoder
from src.hsp.dynamics import ResidualDynamics


class TimeSeriesWindowDataset(Dataset):
    """
    Sliding-window dataset over one or multiple trajectories.

    Each sample provides:
        window: (W, obs_dim) -- input to encoder
        targets: (K, obs_dim) -- next K observations for multi-step loss
        obs_current: (obs_dim,) -- current observation for reconstruction

    Args:
        trajectories: List of 1D arrays (T,) or 2D arrays (T, d).
        window_size: Encoder input window length W.
        prediction_horizon: Number of future steps K for multi-step loss.
    """

    def __init__(
        self,
        trajectories: list[np.ndarray],
        window_size: int = 25,
        prediction_horizon: int = 10,
    ):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.windows = []
        self.targets = []
        self.obs_current = []

        for traj in trajectories:
            if traj.ndim == 1:
                traj = traj[:, None]  # (T,) -> (T, 1)
            T, d = traj.shape
            max_start = T - window_size - prediction_horizon
            for t in range(max_start):
                w = traj[t : t + window_size]            # (W, d)
                tgt = traj[t + window_size : t + window_size + prediction_horizon]  # (K, d)
                obs = traj[t + window_size - 1]           # (d,) last obs in window
                self.windows.append(w.astype(np.float32))
                self.targets.append(tgt.astype(np.float32))
                self.obs_current.append(obs.astype(np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.obs_current[idx]),
        )


class HSPLearnedPipeline(nn.Module):
    """
    End-to-end learned pipeline: encoder + dynamics + decoder.

    Args:
        obs_dim: Observation dimensionality (1 for 1D time series).
        latent_dim: Latent space dimension.
        encoder_hidden: LSTM hidden size.
        encoder_layers: Number of LSTM layers.
        dynamics_hidden: Dynamics MLP hidden size.
        dynamics_layers: Number of dynamics hidden layers.
    """

    def __init__(
        self,
        obs_dim: int = 1,
        latent_dim: int = 8,
        encoder_hidden: int = 32,
        encoder_layers: int = 2,
        dynamics_hidden: int = 32,
        dynamics_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = LSTMEncoder(
            input_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=0.0,
        )
        self.dynamics = ResidualDynamics(
            latent_dim=latent_dim,
            hidden_dim=dynamics_hidden,
            num_layers=dynamics_layers,
        )
        self.decoder = nn.Linear(latent_dim, obs_dim)

    def forward(self, windows: torch.Tensor, K: int = 10):
        """
        Forward pass for training.

        Args:
            windows: (batch, W, obs_dim)
            K: Number of prediction steps.

        Returns:
            z_current: (batch, latent_dim)
            z_predictions: list of K tensors, each (batch, latent_dim)
            x_reconstructed: (batch, obs_dim)
        """
        z = self.encoder(windows)           # (batch, latent_dim)
        x_recon = self.decoder(z)           # (batch, obs_dim)

        # Multi-step rollout
        z_preds = []
        z_step = z
        for _ in range(K):
            z_step = self.dynamics(z_step)  # (batch, latent_dim)
            z_preds.append(z_step)

        return z, z_preds, x_recon


class HSPTrainer:
    """
    Training loop for the HSP learned pipeline.

    Args:
        lr: Learning rate.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        patience: Early stopping patience.
        prediction_horizon: K steps for multi-step loss.
        decay_factor: Geometric decay gamma for multi-step loss.
        recon_weight: Weight lambda for reconstruction loss.
        grad_clip: Gradient clipping norm.
        window_size: Encoder window size.
        device: Torch device.
    """

    def __init__(
        self,
        lr: float = 5e-4,
        epochs: int = 300,
        batch_size: int = 64,
        patience: int = 30,
        prediction_horizon: int = 10,
        decay_factor: float = 0.9,
        recon_weight: float = 0.1,
        grad_clip: float = 1.0,
        window_size: int = 25,
        device: str = "cpu",
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.K = prediction_horizon
        self.gamma = decay_factor
        self.recon_weight = recon_weight
        self.grad_clip = grad_clip
        self.window_size = window_size
        self.device = torch.device(device)

    def fit(
        self,
        pipeline: HSPLearnedPipeline,
        train_trajectories: list[np.ndarray],
        val_trajectories: list[np.ndarray] | None = None,
    ) -> dict:
        """
        Train the pipeline on trajectory data.

        Args:
            pipeline: HSPLearnedPipeline to train.
            train_trajectories: List of raw trajectories (T,) or (T, d).
            val_trajectories: Optional validation trajectories.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        pipeline = pipeline.to(self.device)

        # Create datasets
        train_ds = TimeSeriesWindowDataset(
            train_trajectories, self.window_size, self.K
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_loader = None
        if val_trajectories:
            val_ds = TimeSeriesWindowDataset(
                val_trajectories, self.window_size, self.K
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            pipeline.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            # --- Train ---
            pipeline.train()
            epoch_loss = 0.0
            n_batches = 0

            for windows, targets, obs_curr in train_loader:
                windows = windows.to(self.device)
                targets = targets.to(self.device)
                obs_curr = obs_curr.to(self.device)

                z_curr, z_preds, x_recon = pipeline(windows, K=self.K)

                loss = self._compute_loss(
                    pipeline, z_curr, z_preds, x_recon, windows, targets, obs_curr
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            # --- Val ---
            if val_loader is not None:
                pipeline.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for windows, targets, obs_curr in val_loader:
                        windows = windows.to(self.device)
                        targets = targets.to(self.device)
                        obs_curr = obs_curr.to(self.device)
                        z_curr, z_preds, x_recon = pipeline(windows, K=self.K)
                        loss = self._compute_loss(
                            pipeline, z_curr, z_preds, x_recon, windows, targets, obs_curr
                        )
                        val_loss += loss.item()
                        n_val += 1

                avg_val = val_loss / max(n_val, 1)
                history["val_loss"].append(avg_val)

                # Early stopping
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    wait = 0
                    best_state = {k: v.cpu().clone() for k, v in pipeline.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        if best_state is not None:
                            pipeline.load_state_dict(best_state)
                        break
            else:
                history["val_loss"].append(avg_train)

        # Restore best
        if best_state is not None:
            pipeline.load_state_dict(best_state)
        pipeline = pipeline.to(self.device)

        return history

    def _compute_loss(
        self,
        pipeline: HSPLearnedPipeline,
        z_curr: torch.Tensor,
        z_preds: list[torch.Tensor],
        x_recon: torch.Tensor,
        windows: torch.Tensor,
        targets: torch.Tensor,
        obs_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined multi-step prediction + reconstruction loss.

        L = sum_{k=1}^K gamma^{k-1} MSE(z_hat_{t+k}, z_{t+k}) + lambda * MSE(x_hat, x_t)
        """
        # Encode target observations to get ground-truth latent z_{t+k}
        # For each k, the target window is shifted by k steps.
        # We encode targets by creating shifted windows:
        # target_k has obs at position (t+k), so we directly compare in obs space
        # via the decoder to avoid the need to re-encode.

        # Multi-step loss in observation space (more stable than latent):
        L_multistep = torch.tensor(0.0, device=z_curr.device)
        for k in range(min(self.K, targets.shape[1])):
            x_pred_k = pipeline.decoder(z_preds[k])  # (batch, obs_dim)
            x_true_k = targets[:, k, :]               # (batch, obs_dim)
            L_multistep = L_multistep + (self.gamma ** k) * nn.functional.mse_loss(x_pred_k, x_true_k)

        # Reconstruction loss
        L_recon = nn.functional.mse_loss(x_recon, obs_curr)

        return L_multistep + self.recon_weight * L_recon

    @torch.no_grad()
    def encode_trajectory(
        self,
        pipeline: HSPLearnedPipeline,
        trajectory: np.ndarray,
    ) -> torch.Tensor:
        """
        Encode a full trajectory into latent states using sliding windows.

        Args:
            pipeline: Trained HSPLearnedPipeline.
            trajectory: Raw trajectory, shape (T,) or (T, d).

        Returns:
            z_sequence: (T - window_size + 1, latent_dim) tensor.
        """
        pipeline.eval()
        if trajectory.ndim == 1:
            trajectory = trajectory[:, None]  # (T, 1)

        T, d = trajectory.shape
        W = self.window_size
        n_windows = T - W + 1

        z_list = []
        # Process in batches for efficiency
        batch = 256
        for start in range(0, n_windows, batch):
            end = min(start + batch, n_windows)
            windows = []
            for i in range(start, end):
                windows.append(trajectory[i : i + W])  # (W, d)
            windows_t = torch.tensor(
                np.stack(windows), dtype=torch.float32, device=self.device
            )
            z_batch = pipeline.encoder(windows_t)  # (B, latent_dim)
            z_list.append(z_batch.cpu())

        return torch.cat(z_list, dim=0)  # (n_windows, latent_dim)
