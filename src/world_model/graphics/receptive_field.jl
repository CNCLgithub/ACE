# ==============================================================================
# Pure Julia Receptive Field Geometry & Forward Model
# ==============================================================================

struct ReceptiveField
    x::Float32
    y::Float32
    r::Float32
end

struct RFGeometry
    fields::Vector{ReceptiveField}
end

"""
    generate_receptive_fields(image_shape; ...)

Generates concentric foveal + peripheral receptive field tiling with a fixed
outer visual field radius across arbitrary `target_rf_count` values.
"""
function generate_receptive_fields(image_shape::Tuple{Int, Int};
                                   overlap_density::Float64 = 1.0,
                                   target_rf_count::Int = 128,
                                   fovea_radius_ratio::Float64 = 0.03,
                                   fovea_rf_fraction::Float64 = 0.45,
                                   max_radius_ratio::Float64 = 0.5)::RFGeometry
    w, h = image_shape
    diag = sqrt(Float64(w^2 + h^2))
    r_fovea = Float64(diag * fovea_radius_ratio)
    r_max   = Float64(diag * max_radius_ratio)

    n_fovea = max(1, round(Int, target_rf_count * fovea_rf_fraction))
    n_periph = max(1, target_rf_count - n_fovea)

    fields = ReceptiveField[]

    # -------------------------------------------------------------
    # 1. FOVEA: Uniform, non-overlapping concentric circle packing
    # -------------------------------------------------------------
    ring_counts = Int[1]
    k = 1
    while sum(ring_counts) < n_fovea
        c_k = floor(Int, π / asin(1.0 / (2.0 * k)))
        push!(ring_counts, c_k)
        k += 1
    end
    n_rings_fovea = length(ring_counts) - 1
    ring_counts[end] -= (sum(ring_counts) - n_fovea)

    rf_fovea_r = Float32(r_fovea / (2.0 * n_rings_fovea + 1.0))
    push!(fields, ReceptiveField(0.0f0, 0.0f0, Float32(rf_fovea_r * overlap_density)))

    for ring_idx in 2:length(ring_counts)
        c_k = ring_counts[ring_idx]
        dist_k = Float32(2.0 * (ring_idx - 1) * rf_fovea_r)
        for j in 0:(c_k - 1)
            theta = Float32(j * (2.0 * π / c_k))
            push!(fields, ReceptiveField(dist_k * cos(theta),
                                         dist_k * sin(theta),
                                         Float32(rf_fovea_r * overlap_density)))
        end
    end

    # -------------------------------------------------------------
    # 2. PERIPHERY: Log-Polar Rings Bounded to [r_fovea, r_max]
    # -------------------------------------------------------------
    ratio = r_max / r_fovea

    # Solve for optimal number of rings to span [r_fovea, r_max] given n_periph
    best_n_rings = 1
    best_diff = 1e9
    for nr in 1:60
        rho_cand = ratio^(1.0 / nr)
        s_cand = (rho_cand - 1.0) / (rho_cand + 1.0)
        (s_cand >= 1.0 || s_cand <= 0.0) && continue
        c_est = π / asin(s_cand)
        diff = abs(nr * c_est - n_periph)
        if diff < best_diff
            best_diff = diff
            best_n_rings = nr
        end
    end

    n_rings_periph = best_n_rings
    rho = ratio^(1.0 / n_rings_periph)

    # Distribute n_periph across rings
    base_count = div(n_periph, n_rings_periph)
    rem_count = rem(n_periph, n_rings_periph)
    periph_counts = [base_count + (i <= rem_count ? 1 : 0) for i in 1:n_rings_periph]

    # Generate peripheral rings
    for (i, count) in enumerate(periph_counts)
        inner_r = r_fovea * (rho^(i - 1))
        outer_r = r_fovea * (rho^i)

        # Center distance and radius for exact adjacent ring contact
        d_i = Float32(0.5 * (inner_r + outer_r))
        rf_r = Float32(0.5 * (outer_r - inner_r) * overlap_density)

        offset = (i % 2 == 0) ? Float32(π / count) : 0.0f0

        for j in 0:(count - 1)
            theta = offset + Float32(j * (2.0 * π / count))
            push!(fields, ReceptiveField(d_i * cos(theta), d_i * sin(theta), rf_r))
        end
    end

    return RFGeometry(fields)
end

"""
    circle_circle_intersection(x1, y1, r1, x2, y2, r2) -> Float32

Exact analytical intersection area between two circles.
"""
@inline function circle_circle_intersection(x1::Float32, y1::Float32, r1::Float32,
                                            x2::Float32, y2::Float32, r2::Float32)::Float32
    d = hypot(x2 - x1, y2 - y1)

    # Case 1: Disjoint
    (d >= r1 + r2) && return 0.0f0

    # Case 2: One circle fully inside the other
    if d <= abs(r1 - r2)
        r_min = min(r1, r2)
        return Float32(π * r_min^2)
    end

    # Case 3: Lens overlap
    r1_sq = r1 * r1
    r2_sq = r2 * r2
    d_sq = d * d

    eps_val = 1.0f-8
    alpha = acos(clamp((d_sq + r1_sq - r2_sq) / (2.0f0 * d * r1 + eps_val), -1.0f0, 1.0f0))
    beta  = acos(clamp((d_sq + r2_sq - r1_sq) / (2.0f0 * d * r2 + eps_val), -1.0f0, 1.0f0))
    term  = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)

    return r1_sq * alpha + r2_sq * beta - 0.5f0 * sqrt(max(0.0f0, term))
end

"""
    predict_rf_stats!(means, vars, fixation, geom, objects)

In-place evaluation of receptive field Gaussian RGB statistics.
Zero heap allocations.
"""
function predict_rf_stats!(means::Matrix{Float32},
                           vars::Matrix{Float32},
                           mu_rgb::MVector{3, Float32},
                           var_rgb::MVector{3, Float32},
                           fixation::SVector{2, Float32},
                           geom::RFGeometry,
                           objects::AbstractVector{<:Disc})
    n_rf = length(geom.fields)
    n_obj = length(objects)
    bg_color = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
    # Default object color (1, 1, 1)
    obj_color = SVector{3, Float32}(1.0f0, 1.0f0, 1.0f0)


    covered_area = 0.0f0

    @inbounds for i in 1:n_rf
        rf = geom.fields[i]
        rf_x = rf.x + fixation[1]
        rf_y = rf.y + fixation[2]
        rf_r = rf.r
        rf_area = Float32(π * rf_r^2)

        covered_area = 0.0f0
        fill!(mu_rgb, 0.0f0)
        fill!(var_rgb, 0.0f0)

        for j in 1:n_obj
            obj = objects[j]
            # Disc coordinates and equivalent radius
            overlap = circle_circle_intersection(rf_x, rf_y, rf_r,
                                                 Float32(obj.pos[1]), Float32(obj.pos[2]), Float32(obj.radius))
            covered_area += overlap
            mu_rgb += overlap * obj_color
        end

        bg_area = max(0.0f0, rf_area - covered_area)
        mu_rgb += bg_area * bg_color

        mu_rgb *= 1.0 / rf_area

        # Compute variance across covered objects and background
        for j in 1:n_obj
            obj = objects[j]
            overlap = circle_circle_intersection(rf_x, rf_y, rf_r,
                                                 Float32(obj.pos[1]), Float32(obj.pos[2]), Float32(obj.radius))
            diff = obj_color - mu_rgb
            var_rgb += overlap * (diff .* diff)
        end
        diff_bg = bg_color - mu_rgb
        var_rgb = (var_rgb + bg_area * (diff_bg .* diff_bg)) / rf_area

        for c in 1:3
            means[i, c] = mu_rgb[c]
            vars[i, c] = max(var_rgb[c], 1.0f0) # Minimum variance floor
        end
    end
    return nothing
end

const log_2pi = Float32(log(2.0 * π))

"""
    rf_logpdf(obs, means, vars) -> Float64
"""
function rf_logpdf(obs::AbstractMatrix{Float32}, means::Matrix{Float32}, vars::Matrix{Float32})::Float64
    n_rf, n_channels = size(obs)
    total_logpdf = 0.0f0

    @inbounds for c in 1:n_channels
        for i in 1:n_rf
            diff = obs[i, c] - means[i, c]
            v = vars[i, c]
            lp = -0.5f0 * (log_2pi + log(v) + (diff * diff) / v)
            # if lp < 0
            #     @show diff
            #     @show v
            #     @show lp
            # end
            total_logpdf += lp
        end
    end
    Float64(total_logpdf)
    # result = Float64(total_logpdf)
    # @printf "logpdf %.2f \n" result
    # # error()
    # result
end
