import numpy as np
from scipy.signal import correlate


class EarlyWarningSignals:
    def __init__(self, window: int):
        self.window = window

    def variance(self, series: np.ndarray):
        raise NotImplementedError

    def autocorrelation(self, series: np.ndarray, lag: int = 1):
        raise NotImplementedError

    def critical_slowing_down(self, series: np.ndarray):
        raise NotImplementedError