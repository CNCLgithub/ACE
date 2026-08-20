import numpy as np

import jax
import jax.numpy as jnp
from jax.random import normal, key
from jax import jit, grad, vmap

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
# Interface
################################################################################

def sync_and_sample(fixation, 
                    fields,
                    scene_buf, 
                    n_points: int,
                    seed: int):
    """Bridge function called from Julia."""
    # Zero-copy reinterpret flat C-contiguous buffers from Julia
    fixation_np = np.frombuffer(fixation, dtype=np.float32)
    objects_np = np.frombuffer(scene_buf, dtype=np.float32).reshape((n_points, 7))

    # Transfer into JAX device arrays
    fixation = jax.device_put(fixation_np)
    objects = jax.device_put(objects_np)

    mean, var = predict_rf_stats(fixation, fields, objects)
    # REVIEW: should probably pass key as arg
    sample = (normal(key(seed), mean.shape) + mean) * var
    return np.asarray(sample) # Need to detach to prevent overwriting trace


def sync_and_logpdf(observed_rgb, fixation, fields, objects, n : int) -> float :
    '''
    MAIN API
    Compute the log probability of observed RF colors
    given a scene.
    '''
    # Zero-copy reinterpret flat C-contiguous buffers from Julia
    fixation_np = np.frombuffer(fixation, dtype=np.float32)
    objects_np = np.frombuffer(objects, dtype=np.float32).reshape((n, 7))

    # Transfer into JAX device arrays
    fixation = jax.device_put(fixation_np)
    objects = jax.device_put(objects_np)
    observed_rgb = np.asarray(observed_rgb)

    # xs = np.frombuffer(observed_rgb, dtype=np.float32)
    means, variances = predict_rf_stats(fixation, fields, objects)
    variances = jnp.maximum(variances, 1e-6)
    log_probs = (-0.5 * jnp.log(2.0 * jnp.pi * variances) - 0.5 * ((observed_rgb - means) ** 2) / variances)
    return jnp.sum(log_probs).item()

################################################################################
# Receptive fields - forward function
################################################################################

def generate_receptive_fields(
    image_shape,
    overlap_density=None,
    target_rf_count=128,
    fovea_radius_ratio=0.03, # Fovea radius % of max distance from center
    fovea_rf_fraction=0.45,  # 45% of all RFs allocated to fovea
):
    """
    1. Fovea: Uniformly packed, non-overlapping concentric rings.
    2. Periphery / Parafovea: Successive concentric rings tangent to the foveal outer
       boundary and each other with zero radial gap and minimal overlap.
    """
    height, width = image_shape[:2]
    half_w, half_h = 0.5 * width, 0.5 * height
    max_dist = np.sqrt(half_w**2 + half_h**2)

    r_fovea = max_dist * fovea_radius_ratio
    n_fovea = int(np.round(target_rf_count * fovea_rf_fraction))
    n_periph = target_rf_count - n_fovea

    fields = []

    # -------------------------------------------------------------
    # 1. FOVEA: Uniform, non-overlapping circle packing
    # -------------------------------------------------------------
    ring_counts = [1]
    k = 1
    while sum(ring_counts) < n_fovea:
        c_k = int(np.floor(np.pi / np.arcsin(1.0 / (2.0 * k))))
        ring_counts.append(c_k)
        k += 1

    n_rings_fovea = len(ring_counts) - 1
    ring_counts[-1] -= (sum(ring_counts) - n_fovea)

    # Tangent radius inside fovea
    rf_fovea_r = r_fovea / (2.0 * n_rings_fovea + 1.0)

    # Center RF
    fields.append(np.array([0.0, 0.0, rf_fovea_r]))

    # Concentric foveal rings
    for k in range(1, len(ring_counts)):
        c_k = ring_counts[k]
        dist_k = 2.0 * k * rf_fovea_r
        for j in range(c_k):
            theta = j * (2.0 * np.pi / c_k)
            fields.append(np.array([dist_k * np.cos(theta), dist_k * np.sin(theta), rf_fovea_r]))

    # Outer radius of the foveal region
    r_fovea_outer = r_fovea

    # -------------------------------------------------------------
    # 2. PARAFOVEA & PERIPHERY: Continuous radial progression (No Gaps)
    # -------------------------------------------------------------
    # Distribute remaining RFs across ~5 concentric expanding rings
    n_periph_rings = 5
    periph_counts = [int(np.round(n_periph / n_periph_rings))] * n_periph_rings
    periph_counts[-1] += n_periph - sum(periph_counts)

    # First peripheral ring touches the foveal outer boundary exactly:
    # d_0 - r_0 = r_fovea_outer  =>  d_0 * (1 - sin(pi / count)) = r_fovea_outer
    s_0 = np.sin(np.pi / periph_counts[0])
    curr_d = r_fovea_outer / (1.0 - s_0)

    for i, count in enumerate(periph_counts):
        s_i = np.sin(np.pi / count)
        rf_r = curr_d * s_i

        offset = (np.pi / count) if (i % 2 == 1) else 0.0

        for j in range(count):
            theta = offset + j * (2.0 * np.pi / count)
            fields.append(np.array([curr_d * np.cos(theta), curr_d * np.sin(theta), rf_r]))

        # Tangent step to next ring: next inner boundary touches current outer boundary:
        # d_{next} - r_{next} = d_{curr} + r_{curr}
        if i < len(periph_counts) - 1:
            next_count = periph_counts[i + 1]
            s_next = np.sin(np.pi / next_count)
            curr_d = (curr_d * (1.0 + s_i)) / (1.0 - s_next)

    return jnp.array(fields)

def circle_circle_intersection(x1, y1, r1, x2, y2, r2):
    # Area of intersection between two circles.
    # Returns the area of the portion of the FIRST circle overlapped
    # by the second circle.
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Case 1 : no overlap 
    no_overlap = d >= (r1 + r2)
    # Case 2 : one inside the other 
    contained = d <= jnp.abs(r1 - r2)
    contained_area = jnp.where(r1 <= r2, jnp.pi * r1 ** 2, jnp.pi * r2 ** 2)

    # General Case
    eps = 1e-8
    alpha = jnp.arccos(jnp.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1 + eps), -1.0, 1.0))
    beta = jnp.arccos(jnp.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2 + eps), -1.0, 1.0))
    term = ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    lens_area = (r1 ** 2 * alpha + r2 ** 2 * beta - 0.5 * jnp.sqrt(jnp.maximum(term, 0.0)))

    return jnp.where(no_overlap, 0.0, jnp.where(contained, contained_area, lens_area))

@jit
def predict_rf_stats(fixation, rfs, objects):
    '''
    MAIN FUNCTION
     Args:
        rfs: (N_RF,3) -> [x, y, r]
    objects: (N_OBJ,7) -> [x, y, size, type, r, g, b]
    '''
    obj_x, obj_y, obj_size, obj_type, obj_colors = objects[:, 0], objects[:, 1], objects[:, 2], objects[:, 3], objects[:, 4:8]

    background_color = jnp.array([0.0, 0.0, 0.0])

    # Equivalent radii
    circle_radius = obj_size
    square_radius = jnp.sqrt((obj_size ** 2) / jnp.pi)
    triangle_radius = jnp.sqrt(((jnp.sqrt(3.0) / 4.0) * obj_size ** 2) / jnp.pi)
    eq_radii = jnp.where(obj_type == 0, circle_radius, jnp.where(obj_type == 1, square_radius, triangle_radius))

    fix_x, fix_y = fixation
    
    def compute_single_rf(rf_row):
        rf_x, rf_y, rf_r = rf_row
        rf_x, rf_y = rf_x + fix_x, rf_y + fix_y
        rf_area = jnp.pi * rf_r ** 2
        overlaps = vmap(circle_circle_intersection)(jnp.full_like(obj_x, rf_x),
                                                    jnp.full_like(obj_y, rf_y),
                                                    jnp.full_like(eq_radii, rf_r),
                                                    obj_x, obj_y, eq_radii)
        covered_area = jnp.sum(overlaps)
        background_area = jnp.maximum(0.0, rf_area - covered_area)

        mean_rgb = (jnp.sum(overlaps[:, None] * obj_colors, axis=0) + background_area * background_color) / rf_area
        var_rgb = (jnp.sum(overlaps[:, None] * (obj_colors - mean_rgb) ** 2, axis=0) + background_area * (background_color - mean_rgb) ** 2) / rf_area

        return mean_rgb, jnp.maximum(var_rgb, 1e-6)

    means, variances = vmap(compute_single_rf)(rfs)
    return means, variances


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
