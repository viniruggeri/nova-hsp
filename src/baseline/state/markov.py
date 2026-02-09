"""
Markov Chain Model for state-based collapse prediction.

Discretizes continuous states via clustering, learns transition matrix,
and predicts time to reaching absorbing/collapse states.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MarkovChainModel(BaseEstimator):
    """
    Discrete-time Markov Chain for collapse prediction.
    
    Pipeline:
    1. Discretize continuous states → discrete labels via KMeans
    2. Estimate transition matrix P[i,j] = P(s_t+1=j | s_t=i)
    3. Identify collapse states (from training data)
    4. Simulate chain until hitting collapse state
    
    Usage:
        model = MarkovChainModel(n_states=5)
        model.fit(X_train, T_train)
        t_pred = model.predict_time_to_collapse(X_test)
    """

    def __init__(
        self, 
        n_states: int = 5,
        collapse_quantile: float = 0.2,
        normalize: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_states: Number of discrete states for clustering
            collapse_quantile: Quantile to define collapse states (lower = earlier collapse)
            normalize: Whether to standardize features before clustering
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.collapse_quantile = collapse_quantile
        self.normalize = normalize
        self.random_state = random_state
        
        self.kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
        self.scaler = StandardScaler() if normalize else None
        self.transition_matrix_ = None
        self.collapse_states_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, T: np.ndarray):
        """
        Fit Markov chain from sequences.
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event for each sample (n_samples,)
        
        Returns:
            self
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, n_features), got shape {X.shape}")
        
        n_samples, seq_len, n_features = X.shape
        
        # Flatten all timesteps for clustering
        X_flat = X.reshape(-1, n_features)  # (n_samples * seq_len, n_features)
        
        # Normalize if requested
        if self.normalize:
            X_flat = self.scaler.fit_transform(X_flat)
        
        # Cluster states
        logger.info(f"Clustering {len(X_flat)} timesteps into {self.n_states} states...")
        state_labels = self.kmeans.fit_predict(X_flat)
        
        # Reshape back to sequences
        state_sequences = state_labels.reshape(n_samples, seq_len)  # (n_samples, seq_len)
        
        # Estimate transition matrix
        self.transition_matrix_ = self._estimate_transitions(state_sequences)
        
        # Identify collapse states
        self.collapse_states_ = self._identify_collapse_states(state_sequences, T)
        
        self.is_fitted_ = True
        
        logger.info(f"Markov Chain fitted: {self.n_states} states, "
                   f"{len(self.collapse_states_)} collapse states, "
                   f"transition matrix sparsity: {(self.transition_matrix_ == 0).mean():.2%}")
        
        return self

    def _estimate_transitions(self, state_sequences: np.ndarray) -> np.ndarray:
        """
        Estimate transition matrix from state sequences.
        
        P[i,j] = count(i→j) / count(i)
        """
        counts = np.zeros((self.n_states, self.n_states))
        
        for seq in state_sequences:
            for t in range(len(seq) - 1):
                i, j = seq[t], seq[t+1]
                counts[i, j] += 1
        
        # Normalize rows (add small epsilon to avoid division by zero)
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-10
        transition_matrix = counts / row_sums
        
        return transition_matrix

    def _identify_collapse_states(self, state_sequences: np.ndarray, T: np.ndarray) -> set:
        """
        Identify which states are associated with near-collapse.
        
        Strategy: States that appear frequently near the end of short-lived sequences.
        """
        # Get final state of each sequence
        final_states = state_sequences[:, -1]
        
        # Find samples with early collapse (low T)
        collapse_threshold = np.quantile(T, self.collapse_quantile)
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
        
        logger.info(f"Identified collapse states: {collapse_states}")
        return collapse_states

    def predict_time_to_collapse(
        self, 
        X: np.ndarray, 
        max_steps: int = 100,
        n_simulations: int = 50
    ) -> np.ndarray:
        """
        Predict time to collapse by simulating Markov chain.
        
        Args:
            X: Test sequences (n_samples, seq_len, n_features) or single (seq_len, n_features)
            max_steps: Maximum simulation steps
            n_simulations: Number of Monte Carlo simulations per sample
        
        Returns:
            predicted_times: (n_samples,) predicted time to collapse
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Handle single sample
        is_single = X.ndim == 2
        if is_single:
            X = X[np.newaxis, :, :]
        
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(n_samples):
            # Get current state (last observation)
            x_last = X[i, -1, :]  # (n_features,)
            
            if self.normalize:
                x_last = self.scaler.transform(x_last.reshape(1, -1))
            
            current_state = self.kmeans.predict(x_last.reshape(1, -1))[0]
            
            # Simulate multiple trajectories
            times = []
            for _ in range(n_simulations):
                t = self._simulate_until_collapse(current_state, max_steps)
                times.append(t)
            
            # Use mean predicted time
            pred_time = np.mean(times)
            predictions.append(pred_time)
        
        predictions = np.array(predictions)
        
        if is_single:
            return predictions[0]
        
        return predictions

    def _simulate_until_collapse(self, start_state: int, max_steps: int) -> int:
        """
        Simulate Markov chain until hitting collapse state.
        
        Returns:
            Number of steps until collapse
        """
        current_state = start_state
        
        for t in range(max_steps):
            # Check if in collapse state
            if current_state in self.collapse_states_:
                return t
            
            # Sample next state from transition probabilities
            probs = self.transition_matrix_[current_state]
            current_state = np.random.choice(self.n_states, p=probs)
        
        # Didn't reach collapse in max_steps
        return max_steps

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution π where π = π * P.
        
        Returns:
            Stationary probabilities for each state
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted")
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix_.T)
        
        # Find eigenvector with eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        return stationary