import numpy as np
from sklearn.base import BaseEstimator


class KaplanMeierModel(BaseEstimator):
    def __init__(self):
        self.survival_function_ = None

    def fit(self, durations, events):
        raise NotImplementedError

    def predict_survival(self, t: float):
        raise NotImplementedError