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

function generate_receptive_fields(image_shape::Tuple{Int, Int};
                                   rf_floor_ratio::Float64 = 0.0078,
                                   ecc_scale_ratio::Float64 = 0.0707,
                                   packing::Float64 = 1.5,
                                   max_radius_ratio::Float64 = 1.0)::RFGeometry
    w, h = image_shape
    diag = sqrt(Float64(w^2 + h^2))
    r_max = Float64(diag * max_radius_ratio)
    b_fov = Float64(diag * rf_floor_ratio)
    ecc_scale = Float64(diag * ecc_scale_ratio)

    field_radius(d) = b_fov * (1.0 + d / ecc_scale)

    fields = ReceptiveField[]
    sizehint!(fields, 512)

    push!(fields, ReceptiveField(0.0f0, 0.0f0, Float32(b_fov)))

    d = 0.0
    while true
        b = field_radius(d)
        d_next = d + packing * b
        if d_next >= r_max
            break
        end
        d = d_next
        b = field_radius(d)
        count = max(1, ceil(Int, 2.0 * π * d / (packing * b)))
        for j in 0:(count - 1)
            theta = Float32(j * (2.0 * π / count))
            x = Float32(d * cos(theta))
            y = Float32(d * sin(theta))
            push!(fields, ReceptiveField(x, y, Float32(b)))
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
            means[i, c] = mu_rgb[c]
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
