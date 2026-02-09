"""
Deep Hazard Model - Neural network for hazard prediction.

Learns hazard function via neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class DeepHazardModel(nn.Module, BaseEstimator):
    """
    Deep Hazard Network.
    
    Neural network that predicts time-to-event directly.
    
    Usage:
        model = DeepHazardModel(input_dim=20, hidden_dim=64)
        model.fit(X_train, T_train)
        T_pred = model.predict_time(X_test)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Network architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.hazard_head = nn.Linear(hidden_dim, 1)
        
        self.is_fitted_ = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            predicted_time: (batch_size, 1)
        """
        h = self.encoder(x)
        time_pred = self.hazard_head(h)
        # Use softplus to ensure positive predictions
        time_pred = nn.functional.softplus(time_pred)
        return time_pred

    def fit(self, X: np.ndarray, T: np.ndarray, X_val: np.ndarray = None, T_val: np.ndarray = None,
            epochs: int = 100, batch_size: int = 32, lr: float = 1e-3, device: str = 'cpu'):
        """
        Fit deep hazard model.
        
        Args:
            X: Training features (n_samples, seq_len, n_features) or (n_samples, n_features)
            T: Time-to-event (n_samples,)
            X_val: Validation features (optional)
            T_val: Validation times (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: 'cpu' or 'cuda'
        """
        # Flatten temporal sequences
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)  # (n_samples, seq_len * n_features)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        T_tensor = torch.FloatTensor(T).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, T_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        
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
        logger.info(f"Deep Hazard fitted: {len(X)} samples, best loss={best_loss:.4f}")
        return self

    def predict_time(self, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Predict time-to-event.
        
        Args:
            X: Test features (n_samples, seq_len, n_features) or (n_samples, n_features)
            device: Device for inference
        
        Returns:
            predicted_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Flatten if needed
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.eval()
        self.to(device)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = self.forward(X_tensor)
            predictions = predictions.cpu().numpy().squeeze()
        
        return predictions