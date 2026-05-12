import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional

class TimeConditionedMLP(eqx.Module):
    """
    MLP that takes state `x` and time `t` as input, 
    useful for Vector Fields in Continuous Normalizing Flows (OT-CFM).
    """
    layers: list
    activation: Callable

    def __init__(self, in_size: int, out_size: int, hidden_size: int, depth: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, depth + 1)
        
        # Input to first hidden layer: x (in_size) + t (1)
        self.layers = [eqx.nn.Linear(in_size + 1, hidden_size, key=keys[0])]
        
        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i + 1]))
            
        # Output layer (Vector field outputs same dimension as input state)
        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=keys[-1]))
        
        self.activation = jax.nn.gelu

    def __call__(self, t: float, x: jax.Array) -> jax.Array:
        # Time needs to be broadcasted/concatenated with state
        t_arr = jnp.atleast_1d(t)
        h = jnp.concatenate([x, t_arr], axis=-1)
        
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            
        return self.layers[-1](h)
