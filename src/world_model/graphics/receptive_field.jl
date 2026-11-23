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
    generate_receptive_fields(image_shape; kwargs...) -> RFGeometry

Fixation-centred receptive-field mosaic: an inner foveal block of uniform,
tangent fields, surrounded by `rings_periph` logarithmically spaced rings whose
fields grow and overlap with eccentricity.

Radii are in pixels of `image_shape`. Field centres are relative to the
fixation point, at the origin.

# Arguments

- `image_shape::Tuple{Int,Int}`: `(width, height)`; only the diagonal is used.

# Keyword arguments

- `n_fovea::Int = 58`: number of fields in the foveal block. Rings are packed
  at `c_k = floor(π / asin(1/2k))` fields from the centre outward; the last
  ring is trimmed to land exactly on `n_fovea`.
- `fovea_radius_ratio::Float64 = 0.03`: foveal radius as a fraction of the
  diagonal.
- `max_radius_ratio::Float64 = 1.0`: outer limit `r_max` as a fraction of the
  diagonal. `1.0` means the mosaic reaches the canvas corner from the centre.
- `rings_periph::Int = 8`: number of peripheral rings.
- `tau_max::Float64 = 2.5`: overlap factor at `eccentricity == r_max`. A field
  is `tau` times the half-width of the ring it sits in, so `tau = 1` is
  tangential packing with no overlap and `tau = 2` doubles the radius.
- `overlap_exponent::Float64 = 2.0`: exponent of the overlap schedule.

# Construction

With `rho = (r_max / r_fovea)^(1/rings_periph)`, `s = (rho - 1)/(rho + 1)` and
normalised log eccentricity `ξ = log(d/r_fovea) / log(r_max/r_fovea)`, ring `i`
spanning `[r_fovea rho^(i-1), r_fovea rho^i]` gets

    d_i   = (inner + outer) / 2         ring centre distance
    tau_i = 1 + (tau_max - 1) ξ^q       overlap at the ring centre
    ρ_i   = tau_i (outer - inner) / 2   field radius
    c_i   = ceil(π / arcsin(tau_i s))   fields on the ring

Radial coverage between rings is gaps-free for any `tau_i >= 1`, since
`d_i ± ρ_i` contains `[inner, outer]`. Tangential overlap grows as `tau_i s`
grows; `c_i` falls with eccentricity, so the peripheral field count is not
directly controlled by the caller.

Because `ξ` is normalised by `r_max` while the outermost ring centre lies
inside `r_max`, `tau_max` is approached but not reached: with the defaults,
`tau_8 = 2.34`.

# Returns

`RFGeometry(fields)` with `fields::Vector{ReceptiveField}` of `(x, y, r)`.
Fields are not clipped to the canvas; the outermost ring extends past
`r_max`.

# Example

    geom = generate_receptive_fields((400, 400))
    length(geom.fields)   # 145 = 58 foveal + 87 peripheral
"""
function generate_receptive_fields(image_shape::Tuple{Int, Int};
                                   n_fovea::Int = 58,
                                   fovea_radius_ratio::Float64 = 0.03,
                                   max_radius_ratio::Float64 = 1.0,
                                   rings_periph::Int = 8,
                                   tau_max::Float64 = 2.5,
                                   overlap_exponent::Float64 = 2.0)::RFGeometry
    w, h = image_shape
    diag = sqrt(Float64(w^2 + h^2))
    r_fovea = Float64(diag * fovea_radius_ratio)
    r_max = Float64(diag * max_radius_ratio)

    fields = ReceptiveField[]

    # fovea: tangent concentric rings
    ring_counts = Int[1]
    k = 1
    while sum(ring_counts) < n_fovea
        c_k = floor(Int, π / asin(1.0 / (2.0 * k)))
        push!(ring_counts, c_k)
        k += 1
    end
    n_rings_fovea = length(ring_counts) - 1
    ring_counts[end] -= sum(ring_counts) - n_fovea

    rf_fovea_r = Float32(r_fovea / (2.0 * n_rings_fovea + 1.0))

    push!(fields, ReceptiveField(0.0f0, 0.0f0, rf_fovea_r))

    for ring_idx in 2:length(ring_counts)
        c_k = ring_counts[ring_idx]
        dist_k = Float32(2.0 * (ring_idx - 1) * rf_fovea_r)
        for j in 0:(c_k - 1)
            theta = Float32(j * (2.0 * π / c_k))
            x = dist_k * cos(theta)
            y = dist_k * sin(theta)
            push!(fields, ReceptiveField(x, y, rf_fovea_r))
        end
    end

    # periphery: log-polar rings, radius scaled by the overlap schedule
    rho = (r_max / r_fovea)^(1.0 / rings_periph)
    s = (rho - 1.0) / (rho + 1.0)
    log_span = log(r_max / r_fovea)

    for i in 1:rings_periph
        inner_r = r_fovea * rho^(i - 1)
        outer_r = r_fovea * rho^i
        d_i = 0.5 * (inner_r + outer_r)

        xi = log(d_i / r_fovea) / log_span
        tau = 1.0 + (tau_max - 1.0) * xi^overlap_exponent
        rf_r = Float32(tau * 0.5 * (outer_r - inner_r))

        count = ceil(Int, π / asin(min(tau * s, 0.999)))
        if i % 2 == 0
            offset = Float32(π / count)
        else
            offset = 0.0f0
        end

        for j in 0:(count - 1)
            theta = offset + Float32(j * (2.0 * π / count))
            x = Float32(d_i * cos(theta))
            y = Float32(d_i * sin(theta))
            push!(fields, ReceptiveField(x, y, rf_r))
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
        # for j in 1:n_obj
        #     obj = objects[j]
        #     overlap = circle_circle_intersection(rf_x, rf_y, rf_r,
        #                                          Float32(obj.pos[1]), Float32(obj.pos[2]), Float32(obj.radius))
        #     diff = obj_color - mu_rgb
        #     var_rgb += overlap * (diff .* diff)
        # end
        # diff_bg = bg_color - mu_rgb
        # var_rgb = (var_rgb + bg_area * (diff_bg .* diff_bg)) / rf_area

        for c in 1:3
            # interpreted as psi and kappa of beta dist.
            means[i, c] = clamp(mu_rgb[c], 0.01f0, 0.99f0)
            # vars[i, c] = max(var_rgb[c], 0.01f0) # Minimum variance floor
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
            # normal pdf
            diff = obs[i, c] - means[i, c]
            v = vars[i, c]
            lp = -0.5f0 * (log_2pi + log(v) + (diff * diff) / v)
            total_logpdf += lp
        end
    end
    Float64(total_logpdf)
    # result = Float64(total_logpdf)
    # @printf "logpdf %.2f \n" result
    # # error()
    # result
end

const BETA_LOG = 1f-3
const SIGMA_Y  = 0.4f0
@inline logread(x) = log(x + BETA_LOG)

function rf_logpdf(obs::AbstractMatrix{Float32}, means::Matrix{Float32})::Float64
    ll = 0.0f0
    for i in axes(means, 1), c in 1:3
        x = max(obs[i, c], -BETA_LOG + 1f-6)
        r = logread(x) - logread(means[i, c])
        ll -= 0.5f0 * (r / SIGMA_Y)^2
        ll -= log(sqrt(2f0 * π) * SIGMA_Y)
        ll -= log(x + BETA_LOG)
    end
    return Float64(ll)
end
