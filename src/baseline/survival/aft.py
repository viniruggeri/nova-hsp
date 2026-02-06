import numpy as np
from sklearn.base import BaseEstimator


class AFTModel(BaseEstimator):
    def __init__(self, distribution: str = "weibull"):
        self.distribution = distribution
        self.params_ = None

    def fit(self, X, durations, events):
        raise NotImplementedError

    def predict_time(self, X):
        raise NotImplementedError