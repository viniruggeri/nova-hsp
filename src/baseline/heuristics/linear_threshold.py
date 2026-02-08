import numpy as np


class LinearThresholdHeuristic:
    def __init__(self, weights: np.ndarray, threshold: float, k_steps: int):
        self.weights = weights
        self.threshold = threshold
        self.k_steps = k_steps

    def score(self, X: np.ndarray):
        raise NotImplementedError

    def predict_alert(self, X: np.ndarray):
        raise NotImplementedError