"""
BaseWorld interface for reproducible world simulations.

All world implementations must inherit from BaseWorld and implement:
- __init__(self, cfg): initialize from OmegaConf config
- reset(self, seed: int): reset world state with RNG seed
- step(self): advance simulation by one timestep
- observe(self) -> np.ndarray | dict: return current observations
- is_collapsed(self) -> Tuple[bool, Optional[int]]: check collapse status
- get_true_event_time(self) -> Optional[int]: return collapse timestep if known

Key requirements:
1. Use self.rng = np.random.default_rng(seed) for all randomness
2. Document collapse rule clearly in class docstring
3. Ensure deterministic behavior given the same seed
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict
import numpy as np


class BaseWorld(ABC):
    """
    Abstract base class for world simulations.

    All worlds must be deterministic given a seed and implement
    the core simulation interface defined below.
    """

    def __init__(self, cfg):
        """
        Initialize world from configuration.

        Args:
            cfg: OmegaConf configuration object
        """
        self.cfg = cfg
        self.rng: Optional[np.random.Generator] = None
        self.timestep: int = 0
        self.collapsed: bool = False
        self.collapse_timestep: Optional[int] = None

    @abstractmethod
    def reset(self, seed: int) -> None:
        """
        Reset world to initial state with given seed.

        Must set self.rng = np.random.default_rng(seed) for determinism.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.collapsed = False
        self.collapse_timestep = None

    @abstractmethod
    def step(self) -> None:
        """
        Advance simulation by one timestep.

        Should update internal state and increment self.timestep.
        Must use self.rng for all random operations.
        """
        pass

    @abstractmethod
    def observe(self) -> Union[np.ndarray, Dict[str, float]]:
        """
        Return current observations of the world state.

        Returns:
            Observations as numpy array or dictionary of metrics.
            Should be lightweight and avoid heavy recomputation.
        """
        pass

    @abstractmethod
    def is_collapsed(self) -> Tuple[bool, Optional[int]]:
        """
        Check if world has collapsed.

        Returns:
            Tuple of (collapsed: bool, timestep: Optional[int])
            where timestep is the time of collapse if it occurred.
        """
        pass

    def get_true_event_time(self) -> Optional[int]:
        """
        Return the true collapse/event time if it has occurred.

        Returns:
            Timestep of collapse, or None if not yet collapsed.
        """
        return self.collapse_timestep
