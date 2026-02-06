import numpy as np
from sklearn.base import BaseEstimator


class CoxPHModel(BaseEstimator):
    def __init__(self):
        self.beta_ = None

    def fit(self, X, durations, events):
        raise NotImplementedError

    def predict_risk(self, X):
        raise NotImplementedError

    def predict_survival(self, X, t: float):
        raise NotImplementedError