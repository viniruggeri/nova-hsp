"""
Regression baselines for time-to-event prediction.

Models trained to predict continuous time_to_event instead of binary classification.
This is more natural for survival/event-time data.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge as RidgeRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class RegressionBaselines:
    """Regression models for time-to-event prediction."""

    @staticmethod
    def reshape_for_regression(X: np.ndarray) -> np.ndarray:
        """
        Reshape temporal sequences for regression.

        Input: (n_samples, seq_len, n_features)
        Output: (n_samples, seq_len * n_features) or other aggregations
        """
        n_samples = X.shape[0]

        # Strategy: Use last time step + mean + max + std of sequence
        last_step = X[:, -1, :]  # (n_samples, n_features)
        mean_seq = X.mean(axis=1)  # (n_samples, n_features)
        max_seq = X.max(axis=1)  # (n_samples, n_features)
        std_seq = X.std(axis=1)  # (n_samples, n_features)

        features = np.hstack([last_step, mean_seq, max_seq, std_seq])
        return features

    @staticmethod
    def train_gradient_boosting(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Train Gradient Boosting regressor for time-to-event."""
        try:
            X_train_flat = RegressionBaselines.reshape_for_regression(X_train)
            X_test_flat = RegressionBaselines.reshape_for_regression(X_test)

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)

            # Train
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum(
                (y_test.mean() - y_test) ** 2
            )

            return {
                "predictions": y_pred,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "model": "gradient_boosting",
            }
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
            return None

    @staticmethod
    def train_ridge_regression_time(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Train Ridge regression for time-to-event."""
        try:
            X_train_flat = RegressionBaselines.reshape_for_regression(X_train)
            X_test_flat = RegressionBaselines.reshape_for_regression(X_test)

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)

            # Train
            model = RidgeRegression(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            mape = np.mean(np.abs((y_pred - y_test) / (y_test + 1e-6)))
            r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum(
                (y_test.mean() - y_test) ** 2
            )

            return {
                "predictions": y_pred,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "model": "ridge_regression",
            }
        except Exception as e:
            logger.error(f"Ridge Regression training failed: {e}")
            return None

    @staticmethod
    def train_mlp_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        device="cpu",
        epochs=50,
    ) -> dict:
        """Train MLP neural network for time-to-event prediction."""
        try:
            X_train_flat = RegressionBaselines.reshape_for_regression(X_train)
            X_test_flat = RegressionBaselines.reshape_for_regression(X_test)

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train_scaled).to(device)
            y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

            # Build model
            input_dim = X_train_scaled.shape[1]
            model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # Single output for regression
            ).to(device)

            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_t).cpu().numpy().flatten()
                y_test_np = y_test_t.cpu().numpy().flatten()

            mae = np.mean(np.abs(y_pred - y_test_np))
            rmse = np.sqrt(np.mean((y_pred - y_test_np) ** 2))
            r2 = 1 - np.sum((y_pred - y_test_np) ** 2) / np.sum(
                (y_test_np.mean() - y_test_np) ** 2
            )

            return {
                "predictions": y_pred,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "model": "mlp_regression",
            }
        except Exception as e:
            logger.error(f"MLP Regression training failed: {e}")
            return None
