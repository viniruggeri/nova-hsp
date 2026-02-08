"""
SIR (Susceptible-Infected-Recovered) epidemiological model on a graph.

Collapse Rule:
    The system collapses when EITHER:
    1. Fraction of infected nodes >= collapse_infected_frac, OR
    2. Fraction of susceptible nodes <= collapse_susceptible_frac
    
    This represents epidemic outbreak (too many infected) or 
    complete spread (too few susceptibles remaining).

Model Dynamics:
    - Each node has state: S (Susceptible), I (Infected), or R (Recovered)
    - At each timestep:
        1. Each infected node attempts to infect susceptible neighbors with prob beta
        2. Each infected node recovers with prob gamma
    - Graph structure defined by config (erdos_renyi or barabasi_albert)

Observations:
    Returns dict with:
        - S_frac: fraction of susceptible nodes
        - I_frac: fraction of infected nodes
        - R_frac: fraction of recovered nodes
        - avg_degree: average node degree
        - n_infected_clusters: number of connected components with infected nodes
"""

from typing import Optional, Tuple, Dict
import numpy as np
import networkx as nx
from .base import BaseWorld


class SirGraph(BaseWorld):
    """SIR epidemiological model on a graph network."""

    # State constants
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2

    def __init__(self, cfg):
        """
        Initialize SirGraph world.

        Args:
            cfg: Configuration with fields:
                - population: number of nodes
                - graph.type: 'erdos_renyi' or 'barabasi_albert'
                - graph.p_edge: edge probability (erdos_renyi)
                - graph.m: edges to attach (barabasi_albert)
                - beta: infection probability
                - gamma: recovery probability
                - initial_infected_frac: fraction initially infected
                - collapse.infected_frac: threshold for collapse
                - collapse.susceptible_frac: threshold for collapse
                - max_steps: maximum simulation steps
        """
        super().__init__(cfg)
        self.graph = None
        self.states = None
        self._avg_degree = None

    def reset(self, seed: int) -> None:
        """Reset SIR model with given seed."""
        super().reset(seed)

        # Generate graph based on configuration
        population = self.cfg.population
        graph_type = self.cfg.graph.type

        if graph_type == "erdos_renyi":
            p_edge = self.cfg.graph.p_edge
            self.graph = nx.erdos_renyi_graph(n=population, p=p_edge, seed=seed)
        elif graph_type == "barabasi_albert":
            m = self.cfg.graph.m
            self.graph = nx.barabasi_albert_graph(n=population, m=m, seed=seed)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # Cache average degree (static for this graph)
        self._avg_degree = np.mean([deg for _, deg in self.graph.degree()])

        # Initialize all nodes as susceptible
        self.states = np.zeros(population, dtype=np.int8)

        # Set initial infected nodes
        n_infected = max(1, int(population * self.cfg.initial_infected_frac))
        infected_indices = self.rng.choice(population, size=n_infected, replace=False)
        self.states[infected_indices] = self.INFECTED

    def step(self) -> None:
        """Advance SIR dynamics by one timestep."""
        if self.collapsed:
            return

        new_states = self.states.copy()

        # Process infections: infected nodes try to infect susceptible neighbors
        infected_nodes = np.where(self.states == self.INFECTED)[0]
        for node in infected_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.states[neighbor] == self.SUSCEPTIBLE:
                    # Attempt infection with probability beta
                    if self.rng.random() < self.cfg.beta:
                        new_states[neighbor] = self.INFECTED

        # Process recoveries: infected nodes recover with probability gamma
        for node in infected_nodes:
            if self.rng.random() < self.cfg.gamma:
                new_states[node] = self.RECOVERED

        self.states = new_states
        self.timestep += 1

        # Check for collapse
        collapsed, t = self.is_collapsed()
        if collapsed and not self.collapsed:
            self.collapsed = True
            self.collapse_timestep = t

    def observe(self) -> Dict[str, float]:
        """
        Return current observation of the SIR system.

        Returns:
            Dictionary with state fractions and graph metrics.
        """
        population = len(self.states)

        s_count = np.sum(self.states == self.SUSCEPTIBLE)
        i_count = np.sum(self.states == self.INFECTED)
        r_count = np.sum(self.states == self.RECOVERED)

        # Count infected clusters (connected components with at least one infected)
        infected_nodes = np.where(self.states == self.INFECTED)[0]
        if len(infected_nodes) > 0:
            infected_subgraph = self.graph.subgraph(infected_nodes)
            n_infected_clusters = nx.number_connected_components(infected_subgraph)
        else:
            n_infected_clusters = 0

        return {
            "S_frac": s_count / population,
            "I_frac": i_count / population,
            "R_frac": r_count / population,
            "avg_degree": self._avg_degree,
            "n_infected_clusters": float(n_infected_clusters),
        }

    def is_collapsed(self) -> Tuple[bool, Optional[int]]:
        """
        Check if system has collapsed based on infection/susceptible thresholds.

        Returns:
            (collapsed, timestep) where timestep is when collapse occurred.
        """
        if self.collapsed:
            return True, self.collapse_timestep

        population = len(self.states)
        i_frac = np.sum(self.states == self.INFECTED) / population
        s_frac = np.sum(self.states == self.SUSCEPTIBLE) / population

        # Check collapse conditions
        if i_frac >= self.cfg.collapse.infected_frac:
            return True, self.timestep
        if s_frac <= self.cfg.collapse.susceptible_frac:
            return True, self.timestep

        return False, None
