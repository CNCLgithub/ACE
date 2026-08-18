import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

################################################################################
# TESTS
################################################################################

# tests basic python interopt
def py_hello_world():
    return "Hello, World!"

# tests jax interopt
def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)

norm_compiled = jit(norm)

def test_jax():
    np.random.seed(1701)
    X = jnp.array(np.random.rand(10000, 10))
    return np.allclose(norm(X), norm_compiled(X), atol=1E-6)


################################################################################
# Optimization
################################################################################

@jit
def foveal_coverage_loss(
    fixation: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9
) -> jnp.ndarray:
    """
    Non-parametric expected foveal coverage loss over particle samples.

    Args:
        fixation: Candidate fixation bearing (2,).
        target_samples: Particle trajectories tensor of shape (N_obj, K_particles, H+1, 2).
        weights: Task relevance importance weights w_i(t) of shape (N_obj,).
        sigma_fovea: Foveal radius parameter (free-parameter).
        gamma: Prediction horizon discount factor (free-parameter).
        
    Returns:
        Scalar loss value (negative expected coverage).

    """
    n_obj, k_particles, t_horizon, _ = target_samples.shape

    # 1. Compute the L2 distance between the fixation and object predictions (N, K, H+1)
    # 2. Average the L2 distance per object (N, H+1)
    # 3. Compute the temporal discount of future states using `gamma` (H+1,)
    # 4. Sum over [2] with importance `weights` and temporal discount [3]
    # 5. Return negative total coverate (negative of [4])
    return None

@jit
def movement_cost(
    fixation: jnp.ndarray,
    fixation_prev: jnp.ndarray,
    fixation_vel: jnp.ndarray,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
    eps: float = 1e-6
) -> jnp.ndarray:
    """
    Movement loss function, emphasizes efficient shifts or smooth pursuit.

    Should consider two components:
        1. The L2 distance traveled
        2. Acceleration, or change in velocity.
    
    Args:
        fixation: New fixation (2,).
        fixation_prev: Previous fixation (2,).
        fixation_vel: Previous fixation direction (2,).
        lambda_l2: Cost for amount of movement.
        lambda_smooth: Cost for change in movement.
        eps: Numerical stability for `lambda_l2`.
        
    Returns:
        Scalar loss value.

    """
    pass


@jit
def total_fixation_loss(
    f: jnp.ndarray,
    v_t: jnp.ndarray,
    f_t: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    l_cov = foveal_coverage_loss(f, target_samples, weights)
    l_mov = movement_cost(f, v_t, f_t)
    return l_cov + l_mov


@jit
def resolve_next_fixation_sgd(
    f_t: jnp.ndarray,
    v_t: jnp.ndarray,
    target_samples: jnp.ndarray,
    task_relevance: jnp.ndarray,
    eta_saccade: float = 0.05,
    lr: float = 200.0,
    momentum: float = 0.9,
    num_steps: int = 100,
    bounds: jnp.ndarray = jnp.array([[-400.0, -400.0], [400.0, 400.0]])
) -> jnp.ndarray:
    """
    Resolves next fixation bearing f_{t+1} using Stochastic Gradient Descent (SGD) with Momentum.
    """
    # 1. Softmax task importance weights
    weights = jax.nn.softmax(task_relevance / tau_importance)
    
    loss_fn = lambda f: total_fixation_loss(
        f, f_t, v_t, target_samples, weights,
        sigma_fovea, gamma, lambda_l1, lambda_l2, lambda_smooth
    )
    grad_fn = grad(loss_fn)
    
    # 2. SGD Optimization Loop with Momentum
    def sgd_step(i, val):
        f, v_momentum = val
        g = grad_fn(f)
        v_next = momentum * v_momentum + lr * g
        f_next = f - v_next
        f_next = jnp.clip(f_next, bounds[0], bounds[1])
        return (f_next, v_next)

    f_opt, _ = jax.lax.fori_loop(0, num_steps, sgd_step, (f_t, jnp.zeros(2)))
    
    # 3. Saccade Execution Thresholding
    # Only shift fixation if loss is gain is high enough
    current_loss = loss_fn(f_t)
    opt_loss = loss_fn(f_opt)
    gain = current_loss - opt_loss
    
    f_next = jnp.where(gain > eta_saccade, f_opt, f_t + v_t)
    return f_next
