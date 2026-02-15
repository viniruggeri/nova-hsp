"""
Temporal Classifier using LSTM/GRU.

Sequence-to-scalar prediction for time-to-event.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class TemporalClassifier(nn.Module, BaseEstimator):
    """
    LSTM-based temporal classifier for time-to-event prediction.

    Usage:
        model = TemporalClassifier(input_dim=5, hidden_dim=64)
        model.fit(X_train, T_train)
        T_pred = model.predict_time(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Output layer
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.is_fitted_ = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences (batch_size, seq_len, input_dim)

        Returns:
            predicted_time: (batch_size, 1)
        """
        # RNN forward
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden*directions)

        # Use last timestep
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden*directions)

        # Predict time
        time_pred = self.fc(last_hidden)  # (batch, 1)

        # Ensure positive
        time_pred = nn.functional.softplus(time_pred)

        return time_pred

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        X_val: np.ndarray = None,
        T_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Fit temporal classifier.

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: 'cpu' or 'cuda'
        """
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_samples, seq_len, n_features), got {X.shape}"
            )

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        T_tensor = torch.FloatTensor(T).unsqueeze(1)

        dataset = TensorDataset(X_tensor, T_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0

            for batch_X, batch_T in loader:
                batch_X = batch_X.to(device)
                batch_T = batch_T.to(device)

                optimizer.zero_grad()
                pred = self.forward(batch_X)
                loss = criterion(pred, batch_T)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        self.is_fitted_ = True
        logger.info(
            f"Temporal Classifier ({self.rnn_type.upper()}) fitted: "
            f"{len(X)} samples, best loss={best_loss:.4f}"
        )
        return self

    def predict_time(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Predict time-to-event.

        Args:
            X: Test sequences (n_samples, seq_len, n_features)
            device: Device for inference

        Returns:
            predicted_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, n_features)")

        self.eval()
        self.to(device)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = self.forward(X_tensor)
            predictions = predictions.cpu().numpy().squeeze()

        return predictions
