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

def sync_and_sample(fixation, fields, scene_buf, n_points: int, seed: int):
    fixation_np = np.frombuffer(fixation, dtype=np.float32)
    objects_np = np.frombuffer(scene_buf, dtype=np.float32).reshape((n_points, 7))

    fixation = jax.device_put(fixation_np)
    objects = jax.device_put(objects_np)

    mean, var = predict_rf_stats(fixation, fields, objects)
    sample = sample_normal(seed, mean, var)
    return np.array(sample) # Detach from JAX device to host NumPy array

def sync_and_logpdf(observed_rgb, fixation, fields, objects, n: int) -> float:
    fixation_np = np.frombuffer(fixation, dtype=np.float32)
    objects_np = np.frombuffer(objects, dtype=np.float32).reshape((n, 7))
    # observed_rgb_np = np.asarray(observed_rgb, dtype=np.float32)

    fixation = jax.device_put(fixation_np)
    objects = jax.device_put(objects_np)
    # observed_rgb = jax.device_put(observed_rgb_np)

    means, variances = predict_rf_stats(fixation, fields, objects)
    variances = jnp.maximum(variances, 0.01)
    return float(normal_logpdf(observed_rgb, means, variances).item())

@jit
def sample_normal(seed: int, mus: jnp.ndarray, var: jnp.ndarray):
    # REVIEW: should probably pass key as arg
    sample = (normal(key(seed), mus.shape) + mus) * var
    return sample
    
@jit
def normal_logpdf(xs : jnp.ndarray, mus : jnp.ndarray, vs : jnp.ndarray):
    log_probs = (-0.5 * jnp.log(2.0 * jnp.pi * vs) - 0.5 * ((xs - mus) ** 2) / vs)
    return jnp.sum(log_probs)

def optimize_fixation(
    fixation,
    fixation_vel,
    target_samples_buf,
    task_relevance_buf,
    n_points: int,
    eta_saccade: float = 0.025,
    lr: float = 10.0,
    momentum: float = 0.9,
    num_steps: int = 100,
    bounds: jnp.ndarray = jnp.array([[-200.0, -200.0], [200.0, 200.0]]),
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
):
    fixation_np = np.frombuffer(fixation, dtype=np.float32)
    fixvel_np = np.frombuffer(fixation_vel, dtype=np.float32)
    targets_np = np.frombuffer(target_samples_buf, dtype=np.float32).reshape((n_points, 2))
    weights_np = np.frombuffer(task_relevance_buf, dtype=np.float32)

    fixation_dev = jax.device_put(fixation_np)
    fixvel_dev = jax.device_put(fixvel_np)
    targets_dev = jax.device_put(targets_np)
    weights_dev = jax.device_put(weights_np)

    new_fixation = resolve_next_fixation_gd(
        fixation_dev,
        fixvel_dev,
        targets_dev,
        weights_dev,
        eta_saccade = eta_saccade,
        lr=lr,
        momentum=momentum,
        num_steps=num_steps,
        bounds=bounds,
        sigma_fovea=sigma_fovea,
        gamma=gamma,
        lambda_l2=lambda_l2,
        lambda_smooth=lambda_smooth
    )
    return np.array(new_fixation)


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
@jit
def foveal_coverage_loss(
    fixation: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
) -> jnp.ndarray:
    """Calculates weighted foveal coverage over target samples."""
    if target_samples.ndim == 2:
        # target_samples: (N_targets, 2)
        distances = jnp.linalg.norm(target_samples - fixation, axis=-1)
        coverage = jnp.exp(-0.5 * (distances / sigma_fovea) ** 2)
        total_coverage = jnp.sum(coverage * weights)
        return -total_coverage

    elif target_samples.ndim == 4:
        # target_samples: (N_obj, K_particles, H+1, 2)
        distances = jnp.linalg.norm(target_samples - fixation, axis=-1)
        coverage = jnp.exp(-0.5 * (distances / sigma_fovea) ** 2)
        expected_coverage = jnp.mean(coverage, axis=1)  # (N_obj, H+1)
        temporal_discount = gamma ** jnp.arange(target_samples.shape[2])
        total_coverage = jnp.sum(expected_coverage * weights[:, None] * temporal_discount)
        return -total_coverage

    else:
        raise ValueError(f"Unexpected target_samples shape: {target_samples.shape}")


@jit
def movement_cost(
    fixation: jnp.ndarray,
    fixation_prev: jnp.ndarray,
    fixation_vel: jnp.ndarray,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Penalize both saccade distance and deviation from prior velocity."""
    if fixation.shape != (2,) or fixation_prev.shape != (2,) or fixation_vel.shape != (2,):
        raise ValueError("fixation, fixation_prev, and fixation_vel must each have shape (2,)")
    displacement = fixation - fixation_prev
    movement = jnp.sqrt(jnp.sum(displacement**2) + eps)
    acceleration = displacement - fixation_vel
    return lambda_l2 * movement + lambda_smooth * jnp.sum(acceleration**2)


@jit
def total_fixation_loss(
    fixation: jnp.ndarray,
    fixation_prev: jnp.ndarray,
    fixation_vel: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
) -> jnp.ndarray:
    """Combined task-relevance and movement objective to minimize."""
    return foveal_coverage_loss(fixation, target_samples, weights, sigma_fovea, gamma) + movement_cost(
        fixation, fixation_prev, fixation_vel, lambda_l2, lambda_smooth
    )


@jit
def resolve_next_fixation_gd(
    f_t: jnp.ndarray,
    v_t: jnp.ndarray,
    target_samples: jnp.ndarray,
    task_relevance: jnp.ndarray,
    eta_saccade: float = 0.025,
    lr: float = 10.0,
    momentum: float = 0.9,
    num_steps: int = 100,
    bounds: jnp.ndarray = jnp.array([[-200.0, -200.0], [200.0, 200.0]]),
    tau_importance: float = 1.0,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
) -> jnp.ndarray:
    """Optimize and threshold the next fixation using momentum SGD.

    ``f_t`` is the current fixation and ``v_t`` is its previous displacement
    (both in pixels per simulation timestep). ``task_relevance`` contains one
    unnormalized importance value per object and is normalized by softmax.
    """
    if f_t.shape != (2,) or v_t.shape != (2,):
        raise ValueError("f_t and v_t must each have shape (2,)")
    if bounds.shape != (2, 2):
        raise ValueError("bounds must have shape (2, 2)")
    if task_relevance.shape != (target_samples.shape[0],):
        raise ValueError("task_relevance must have one entry per object")
    weights = jax.nn.softmax(task_relevance / tau_importance)

    def loss_fn(fixation: jnp.ndarray) -> jnp.ndarray:
        return total_fixation_loss(
            fixation,
            f_t,
            v_t,
            target_samples,
            weights,
            sigma_fovea,
            gamma,
            lambda_l2,
            lambda_smooth,
        )

    grad_fn = grad(loss_fn)

    def gd_step(_: int, value: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        fixation, velocity_momentum = value
        gradient = grad_fn(fixation)
        next_momentum = momentum * velocity_momentum + lr * gradient
        next_fixation = jnp.clip(fixation - next_momentum, bounds[0], bounds[1])
        return next_fixation, next_momentum

    f_opt, _ = jax.lax.fori_loop(0, num_steps, gd_step, (f_t, jnp.zeros(2)))
    gain = loss_fn(f_t) - loss_fn(f_opt)
    # With insufficient improvement, continue smooth pursuit instead of saccading.
    is_smooth = gain > eta_saccade
    return jnp.where(is_smooth, f_opt, f_t + v_t)
