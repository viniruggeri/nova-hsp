"""
Hidden Markov Model for state-based prediction.

Models temporal sequences as transitions between hidden states.
"""

import numpy as np
from sklearn.base import BaseEstimator
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)


class HMMStateModel(BaseEstimator):
    """
    Gaussian Hidden Markov Model.
    
    Models sequences as emissions from hidden states with Gaussian distributions.
    
    Usage:
        model = HMMStateModel(n_states=3)
        model.fit(X_train, T_train)
        T_pred = model.predict_time_to_collapse(X_test)
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, random_state: int = 42):
        """
        Args:
            n_states: Number of hidden states
            n_iter: Maximum EM iterations
            random_state: Random seed
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=n_iter,
            random_state=random_state
        )
        self.collapse_states_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, T: np.ndarray):
        """
        Fit HMM from sequences.
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event for each sample (n_samples,)
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, n_features), got {X.shape}")
        
        n_samples, seq_len, n_features = X.shape
        
        # Concatenate all sequences
        X_concat = X.reshape(-1, n_features)  # (n_samples * seq_len, n_features)
        lengths = [seq_len] * n_samples  # All same length
        
        # Fit HMM
        try:
            self.model.fit(X_concat, lengths)
            logger.info(f"HMM fitted: {n_samples} sequences, {self.n_states} states, "
                       f"converged={self.model.monitor_.converged}")
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            raise
        
        # Identify collapse states
        self._identify_collapse_states(X, T)
        
        self.is_fitted_ = True
        return self

    def _identify_collapse_states(self, X: np.ndarray, T: np.ndarray):
        """
        Identify which hidden states correspond to collapse.
        
        States that appear more frequently near the end of short-lived sequences.
        """
        # Get final hidden state of each sequence
        final_states = []
        for i in range(len(X)):
            states = self.model.predict(X[i])
            final_states.append(states[-1])
        
        final_states = np.array(final_states)
        
        # Find samples with early collapse (bottom 20%)
        collapse_threshold = np.percentile(T, 20)
        early_collapse_mask = T <= collapse_threshold
        
        # States that appear as final states in early-collapse samples
        collapse_states = set(final_states[early_collapse_mask])
        
        if len(collapse_states) == 0:
            # Fallback: use state with lowest average T
            state_to_avg_T = {}
            for state in range(self.n_states):
                mask = final_states == state
                if mask.any():
                    state_to_avg_T[state] = T[mask].mean()
            
            worst_state = min(state_to_avg_T, key=state_to_avg_T.get)
            collapse_states = {worst_state}
        
        self.collapse_states_ = collapse_states
        logger.info(f"Identified collapse states: {collapse_states}")

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hidden state sequence using Viterbi algorithm.
        
        Args:
            X: Sequence (seq_len, n_features)
        
        Returns:
            states: (seq_len,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict(X)

    def predict_state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities at each timestep.
        
        Args:
            X: Sequence (seq_len, n_features)
        
        Returns:
            probabilities: (seq_len, n_states)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)

    def predict_time_to_collapse(self, X: np.ndarray, max_steps: int = 50) -> np.ndarray:
        """
        Predict time until reaching collapse state.
        
        Args:
            X: Test sequences (n_samples, seq_len, n_features) or single (seq_len, n_features)
            max_steps: Maximum forward simulation steps
        
        Returns:
            predicted_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        is_single = X.ndim == 2
        if is_single:
            X = X[np.newaxis, :, :]
        
        predictions = []
        
        for i in range(len(X)):
            # Get current state
            states = self.model.predict(X[i])
            current_state = states[-1]
            
            # Check if already in collapse state
            if current_state in self.collapse_states_:
                predictions.append(0)
                continue
            
            # Simulate forward using transition matrix
            steps = 0
            state = current_state
            
            for _ in range(max_steps):
                if state in self.collapse_states_:
                    break
                
                # Sample next state
                probs = self.model.transmat_[state]
                state = np.random.choice(self.n_states, p=probs)
                steps += 1
            
            predictions.append(steps)
        
        predictions = np.array(predictions)
        
        if is_single:
            return predictions[0]
        
        return predictions