"""
Tests for world implementations.

Tests verify:
- Deterministic behavior (reset with same seed produces same results)
- Observation shape/type consistency
- Collapse detection returns correct types
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from src.worlds.sir_graph import SirGraph
from src.worlds.ant_colony import AntColony


class TestSirGraph:
    """Tests for SirGraph world."""

    @pytest.fixture
    def config(self):
        """Load SirGraph configuration."""
        return OmegaConf.load("configs/worlds/sir_graph.yaml")

    def test_reset_deterministic(self, config):
        """Test that reset with same seed produces deterministic results."""
        world1 = SirGraph(config)
        world2 = SirGraph(config)

        seed = 42
        world1.reset(seed)
        world2.reset(seed)

        # Run a few steps and compare observations
        obs1_list = []
        obs2_list = []

        for _ in range(5):
            obs1 = world1.observe()
            obs2 = world2.observe()
            obs1_list.append(obs1)
            obs2_list.append(obs2)

            world1.step()
            world2.step()

        # Compare observations
        for obs1, obs2 in zip(obs1_list, obs2_list):
            for key in obs1.keys():
                assert np.isclose(
                    obs1[key], obs2[key]
                ), f"Mismatch in {key}: {obs1[key]} vs {obs2[key]}"

    def test_observe_shape(self, config):
        """Test that observe returns consistent dict structure."""
        world = SirGraph(config)
        world.reset(0)

        obs = world.observe()

        # Check it's a dict with expected keys
        assert isinstance(obs, dict)
        expected_keys = {
            "S_frac",
            "I_frac",
            "R_frac",
            "avg_degree",
            "n_infected_clusters",
        }
        assert set(obs.keys()) == expected_keys

        # Check all values are numeric
        for key, value in obs.items():
            assert isinstance(
                value, (int, float, np.number)
            ), f"{key} should be numeric, got {type(value)}"

        # Check fractions sum to ~1.0
        total_frac = obs["S_frac"] + obs["I_frac"] + obs["R_frac"]
        assert np.isclose(total_frac, 1.0, atol=1e-6)

    def test_is_collapsed_returns(self, config):
        """Test that is_collapsed returns correct types."""
        world = SirGraph(config)
        world.reset(0)

        collapsed, timestep = world.is_collapsed()

        # Check types
        assert isinstance(collapsed, bool)
        assert timestep is None or isinstance(timestep, int)

        # If collapsed, timestep should be an int
        if collapsed:
            assert isinstance(timestep, int)

    def test_collapse_occurs(self, config):
        """Test that collapse eventually occurs with aggressive parameters."""
        # Modify config for faster collapse
        config.beta = 0.9  # High infection rate
        config.collapse.infected_frac = 0.3  # Lower threshold

        world = SirGraph(config)
        world.reset(0)

        max_iterations = 100
        collapsed = False

        for _ in range(max_iterations):
            world.step()
            collapsed, _ = world.is_collapsed()
            if collapsed:
                break

        # Should collapse within reasonable time
        assert collapsed, "World should have collapsed with aggressive parameters"


class TestAntColony:
    """Tests for AntColony world."""

    @pytest.fixture
    def config(self):
        """Load AntColony configuration."""
        return OmegaConf.load("configs/worlds/ant_colony.yaml")

    def test_reset_deterministic(self, config):
        """Test that reset with same seed produces deterministic results."""
        world1 = AntColony(config)
        world2 = AntColony(config)

        seed = 42
        world1.reset(seed)
        world2.reset(seed)

        # Run a few steps and compare observations
        obs1_list = []
        obs2_list = []

        for _ in range(5):
            obs1 = world1.observe()
            obs2 = world2.observe()
            obs1_list.append(obs1)
            obs2_list.append(obs2)

            world1.step()
            world2.step()

        # Compare observations
        for obs1, obs2 in zip(obs1_list, obs2_list):
            for key in obs1.keys():
                assert np.isclose(
                    obs1[key], obs2[key]
                ), f"Mismatch in {key}: {obs1[key]} vs {obs2[key]}"

    def test_observe_shape(self, config):
        """Test that observe returns consistent dict structure."""
        world = AntColony(config)
        world.reset(0)

        obs = world.observe()

        # Check it's a dict with expected keys
        assert isinstance(obs, dict)
        expected_keys = {
            "total_resource",
            "avg_resource_per_cell",
            "starving_fraction",
            "avg_ant_degree",
        }
        assert set(obs.keys()) == expected_keys

        # Check all values are numeric
        for key, value in obs.items():
            assert isinstance(
                value, (int, float, np.number)
            ), f"{key} should be numeric, got {type(value)}"

        # Check starving_fraction is in [0, 1]
        assert 0.0 <= obs["starving_fraction"] <= 1.0

    def test_is_collapsed_returns(self, config):
        """Test that is_collapsed returns correct types."""
        world = AntColony(config)
        world.reset(0)

        collapsed, timestep = world.is_collapsed()

        # Check types
        assert isinstance(collapsed, bool)
        assert timestep is None or isinstance(timestep, int)

        # If collapsed, timestep should be an int
        if collapsed:
            assert isinstance(timestep, int)

    def test_collapse_occurs(self, config):
        """Test that collapse eventually occurs with aggressive parameters."""
        # Modify config for faster collapse
        config.consumption_rate = 5.0  # High consumption
        config.replenish_rate = 0.5  # Low replenishment
        config.collapse.resource_threshold = 200.0  # Higher threshold

        world = AntColony(config)
        world.reset(0)

        max_iterations = 100
        collapsed = False

        for _ in range(max_iterations):
            world.step()
            collapsed, _ = world.is_collapsed()
            if collapsed:
                break

        # Should collapse within reasonable time
        assert collapsed, "World should have collapsed with aggressive parameters"


class TestGetTrueEventTime:
    """Test get_true_event_time method for both worlds."""

    def test_sir_event_time(self):
        """Test that SirGraph returns correct event time."""
        config = OmegaConf.load("configs/worlds/sir_graph.yaml")
        world = SirGraph(config)
        world.reset(0)

        # Before collapse
        assert world.get_true_event_time() is None

        # Run until collapse
        for _ in range(200):
            world.step()
            collapsed, _ = world.is_collapsed()
            if collapsed:
                break

        # After collapse
        if collapsed:
            event_time = world.get_true_event_time()
            assert event_time is not None
            assert isinstance(event_time, int)
            assert event_time > 0

    def test_ant_event_time(self):
        """Test that AntColony returns correct event time."""
        config = OmegaConf.load("configs/worlds/ant_colony.yaml")
        world = AntColony(config)
        world.reset(0)

        # Before collapse
        assert world.get_true_event_time() is None

        # Modify for faster collapse
        config.consumption_rate = 5.0
        config.replenish_rate = 0.5
        world = AntColony(config)
        world.reset(0)

        # Run until collapse
        for _ in range(100):
            world.step()
            collapsed, _ = world.is_collapsed()
            if collapsed:
                break

        # After collapse
        if collapsed:
            event_time = world.get_true_event_time()
            assert event_time is not None
            assert isinstance(event_time, int)
            assert event_time > 0
