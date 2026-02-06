import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class TemporalClassifier:
    def __init__(self, model_type: str = "lstm", input_dim: int | None = None):
        self.model_type = model_type

        if model_type == "lstm":
            if input_dim is None:
                raise ValueError("input_dim is required for LSTM")
            self.model = nn.LSTM(input_dim, hidden_size=64, batch_first=True)
            self.head = nn.Linear(64, 1)

        elif model_type == "rf":
            self.model = RandomForestClassifier()

        else:
            raise ValueError("model_type must be 'lstm' or 'rf'")

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError