"""
Agent-Based Model of an Ant Colony with resource dynamics.

Collapse Rule:
    The system collapses when EITHER:
    1. Total resources across all cells <= collapse_resource_threshold, OR
    2. Fraction of starving ants >= collapse_starving_frac
    
    This represents colony collapse due to resource depletion or mass starvation.

Model Dynamics:
    - Grid/graph with n_cells, each having a resource level
    - n_ants distributed across cells, each needing resources to survive
    - At each timestep:
        1. Ants move to neighboring cells with prob movement_prob
        2. Ants consume resources (consumption_rate per ant per cell)
        3. Resources replenish (replenish_rate per cell)
        4. Ants become starving if cell resources < consumption_rate
    
Observations:
    Returns dict with:
        - total_resource: sum of resources across all cells
        - avg_resource_per_cell: mean resource per cell
        - starving_fraction: fraction of ants in cells with insufficient resources
        - avg_ant_degree: average connectivity of ant positions
"""

from typing import Optional, Tuple, Dict
import numpy as np
import networkx as nx
from .base import BaseWorld


class AntColony(BaseWorld):
    """Agent-based ant colony simulation with resource dynamics."""

    def __init__(self, cfg):
        """
        Initialize AntColony world.

        Args:
            cfg: Configuration with fields:
                - n_cells: number of cells/locations
                - n_ants: number of ants
                - graph.type: 'grid' or 'random'
                - graph.grid_size: for grid layout (n_cells = grid_size^2)
                - graph.k_neighbors: for random graph connectivity
                - initial_resource: starting resource per cell
                - consumption_rate: resource consumed per ant per timestep
                - replenish_rate: resource added per cell per timestep
                - movement_prob: probability ant moves to neighbor
                - collapse.resource_threshold: total resource collapse threshold
                - collapse.starving_frac: starving fraction collapse threshold
                - max_steps: maximum simulation steps
        """
        super().__init__(cfg)
        self.graph = None
        self.resources = None  # Resource per cell
        self.ant_positions = None  # Ant locations (cell indices)
        self._avg_degree = None

    def reset(self, seed: int) -> None:
        """Reset ant colony with given seed."""
        super().reset(seed)

        n_cells = self.cfg.n_cells
        n_ants = self.cfg.n_ants

        # Generate graph structure
        graph_type = self.cfg.graph.type

        if graph_type == "grid":
            # Create 2D grid graph
            grid_size = int(np.sqrt(n_cells))
            if grid_size * grid_size != n_cells:
                grid_size = int(np.ceil(np.sqrt(n_cells)))
            self.graph = nx.grid_2d_graph(grid_size, grid_size)
            # Relabel nodes to integers
            self.graph = nx.convert_node_labels_to_integers(self.graph)
            # Adjust n_cells if needed
            if self.graph.number_of_nodes() != n_cells:
                self.cfg.n_cells = self.graph.number_of_nodes()
                n_cells = self.cfg.n_cells

        elif graph_type == "random":
            # Create random regular graph
            k_neighbors = self.cfg.graph.k_neighbors
            # Ensure k < n and n*k is even for random_regular_graph
            k_neighbors = min(k_neighbors, n_cells - 1)
            if n_cells * k_neighbors % 2 != 0:
                k_neighbors = k_neighbors - 1 if k_neighbors > 0 else 0

            if k_neighbors > 0:
                self.graph = nx.random_regular_graph(
                    d=k_neighbors, n=n_cells, seed=seed
                )
            else:
                # Fallback to path graph if k=0
                self.graph = nx.path_graph(n_cells)

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # Cache average degree
        self._avg_degree = np.mean([deg for _, deg in self.graph.degree()])

        # Initialize resources
        self.resources = np.full(n_cells, self.cfg.initial_resource, dtype=np.float32)

        # Distribute ants randomly across cells
        self.ant_positions = self.rng.choice(n_cells, size=n_ants, replace=True)

    def step(self) -> None:
        """Advance ant colony dynamics by one timestep."""
        if self.collapsed:
            return

        n_cells = self.cfg.n_cells
        n_ants = self.cfg.n_ants

        # 1. Ant movement: ants move to neighboring cells with movement_prob
        new_positions = self.ant_positions.copy()
        for i, current_cell in enumerate(self.ant_positions):
            if self.rng.random() < self.cfg.movement_prob:
                # Get neighbors
                neighbors = list(self.graph.neighbors(current_cell))
                if neighbors:
                    new_positions[i] = self.rng.choice(neighbors)

        self.ant_positions = new_positions

        # 2. Resource consumption: ants consume resources from their cells
        for cell in range(n_cells):
            ants_in_cell = np.sum(self.ant_positions == cell)
            consumption = ants_in_cell * self.cfg.consumption_rate
            self.resources[cell] = max(0.0, self.resources[cell] - consumption)

        # 3. Resource replenishment
        self.resources += self.cfg.replenish_rate

        self.timestep += 1

        # Check for collapse
        collapsed, t = self.is_collapsed()
        if collapsed and not self.collapsed:
            self.collapsed = True
            self.collapse_timestep = t

    def observe(self) -> Dict[str, float]:
        """
        Return current observation of the ant colony.

        Returns:
            Dictionary with resource and ant metrics.
        """
        n_cells = self.cfg.n_cells
        n_ants = self.cfg.n_ants

        total_resource = float(np.sum(self.resources))
        avg_resource = float(np.mean(self.resources))

        # Count starving ants (in cells with resource < consumption_rate)
        starving_count = 0
        for ant_pos in self.ant_positions:
            if self.resources[ant_pos] < self.cfg.consumption_rate:
                starving_count += 1

        starving_frac = starving_count / n_ants if n_ants > 0 else 0.0

        # Average degree of cells where ants are located
        ant_unique_positions = np.unique(self.ant_positions)
        if len(ant_unique_positions) > 0:
            ant_degrees = [self.graph.degree(int(pos)) for pos in ant_unique_positions]
            avg_ant_degree = float(np.mean(ant_degrees))
        else:
            avg_ant_degree = 0.0

        return {
            "total_resource": total_resource,
            "avg_resource_per_cell": avg_resource,
            "starving_fraction": starving_frac,
            "avg_ant_degree": avg_ant_degree,
        }

    def is_collapsed(self) -> Tuple[bool, Optional[int]]:
        """
        Check if colony has collapsed based on resource/starvation thresholds.

        Returns:
            (collapsed, timestep) where timestep is when collapse occurred.
        """
        if self.collapsed:
            return True, self.collapse_timestep

        total_resource = np.sum(self.resources)

        # Count starving ants
        starving_count = 0
        for ant_pos in self.ant_positions:
            if self.resources[ant_pos] < self.cfg.consumption_rate:
                starving_count += 1

        starving_frac = starving_count / self.cfg.n_ants if self.cfg.n_ants > 0 else 0.0

        # Check collapse conditions
        if total_resource <= self.cfg.collapse.resource_threshold:
            return True, self.timestep
        if starving_frac >= self.cfg.collapse.starving_frac:
            return True, self.timestep

        return False, None
