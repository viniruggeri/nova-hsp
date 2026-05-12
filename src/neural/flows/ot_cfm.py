import jax
import jax.numpy as jnp
import equinox as eqx

def exact_optimal_transport_conditional_vector_field(x0: jax.Array, x1: jax.Array) -> jax.Array:
    """
    Computes the conditional vector field for Exact Optimal Transport (OT-CFM).
    Because we use independent Gaussian noise (or empirical distributions),
    the map is simply the straight line between x0 and x1.
    
    Args:
        x0: Sample from the source distribution p_0 (e.g., standard Gaussian).
        x1: Sample from the target distribution p_1 (e.g., empirical data).
        
    Returns:
        Vector field u_t(x|x0, x1) which is constant in time for exact OT: x1 - x0.
    """
    return x1 - x0

def sample_conditional_pt(x0: jax.Array, x1: jax.Array, t: float) -> jax.Array:
    """
    Samples from the conditional probability path p_t(x | x0, x1).
    For OT-CFM, this is the deterministic interpolation.
    
    Args:
        x0: Source sample.
        x1: Target sample.
        t: Time t in [0, 1].
        
    Returns:
        x_t = (1 - t) * x0 + t * x1
    """
    return (1.0 - t) * x0 + t * x1

@eqx.filter_jit
def ot_cfm_loss(vector_field_model: eqx.Module, x0: jax.Array, x1: jax.Array, t: float) -> jax.Array:
    """
    Computes the Conditional Flow Matching (CFM) loss for a single pair and time.
    
    Args:
        vector_field_model: The neural network v_theta(t, x).
        x0: Source sample.
        x1: Target sample.
        t: Sampled time t ~ U(0, 1).
        
    Returns:
        L2 loss between the neural vector field and the target conditional vector field.
    """
    # 1. Sample intermediate point
    xt = sample_conditional_pt(x0, x1, t)
    
    # 2. Compute target vector field
    ut = exact_optimal_transport_conditional_vector_field(x0, x1)
    
    # 3. Model prediction
    vt = vector_field_model(t, xt)
    
    # 4. L2 Loss
    return jnp.mean((vt - ut)**2)
