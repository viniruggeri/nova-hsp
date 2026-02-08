import numpy as np


class HMMStateModel:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.transition_matrix_ = None
        self.emission_params_ = None

    def fit(self, signals: np.ndarray):
        raise NotImplementedError

    def predict_states(self, signals: np.ndarray):
        raise NotImplementedError

    def predict_state_probabilities(self, signals: np.ndarray):
        raise NotImplementedError