import numpy as np


class MarkovChainModel:
    def __init__(self, n_states: int):
        self.n_states = n_states
        self.transition_matrix_ = None

    def fit(self, state_sequence: np.ndarray):
        raise NotImplementedError

    def predict_next_state(self, current_state: int):
        raise NotImplementedError