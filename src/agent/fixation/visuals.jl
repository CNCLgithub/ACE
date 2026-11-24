export paint_state, relevance_density, relevance_grid

################################################################################
# Fixation-terrain visualisation
#
# The fixation controller maximises the weighted foveal coverage
#
#     C(f) = sum_i w_i exp(-||x_i - f||^2 / (2 sigma^2))
#
# over gaze position f, where x_i are the task-relevance support points and w_i
# their masses (`attention_map`). The functions below render C over the arena,
# together with the current fixation and the candidate the optimiser proposes,
# so that a fixation which fails to move can be attributed either to the shape
# of C (a perceptual or relevance defect) or to the controller's gate.
################################################################################

"""
    relevance_density(targets, weights, sigma, x, y) -> Float64

Task-relevance density at `(x, y)`:

    R(x, y) = sum_i w_i exp(-||x_i - (x, y)||^2 / (2 sigma^2))

This is the negated loss of `foveal_loss_and_grad!` with the same kernel, so
`argmax R` is the fixation the controller is trying to reach.
"""
function relevance_density(targets::AbstractVector{S2V},
                           weights::AbstractVector{Float64},
                           sigma::Float64,
                           x::Float64,
                           y::Float64)
    inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma)
    density = 0.0
    @inbounds for i in eachindex(targets)
        dx = targets[i][1] - x
        dy = targets[i][2] - y
        density += weights[i] * exp(-(dx * dx + dy * dy) * inv_2sigma_sq)
    end
    return density
end

"""
    relevance_grid(targets, weights, sigma, bounds, n=48) -> Matrix{Float64}

Sample `relevance_density` on an `n x n` lattice over `bounds = (x0, y0, x1, y1)`.
Entry `[i, j]` corresponds to cell centre `(x0 + (i - 0.5) dx, y0 + (j - 0.5) dy)`.
"""
function relevance_grid(targets::AbstractVector{S2V},
                        weights::AbstractVector{Float64},
                        sigma::Float64,
                        bounds::Tuple{Float64, Float64, Float64, Float64},
                        n::Int = 48)
    x0, y0, x1, y1 = bounds
    table = zeros(Float64, n, n)
    cw = (x1 - x0) / n
    ch = (y1 - y0) / n
    @inbounds for j in 1:n
        yc = y0 + (j - 0.5) * ch
        for i in 1:n
            xc = x0 + (i - 0.5) * cw
            table[i, j] = relevance_density(targets, weights, sigma, xc, yc)
        end
    end
    return table
end

"Colour ramp: black -> indigo -> magenta -> orange -> pale yellow."
const HEAT_STOPS = ((0.00, 0.02, 0.02, 0.05),
                    (0.30, 0.16, 0.05, 0.35),
                    (0.55, 0.60, 0.10, 0.45),
                    (0.80, 0.90, 0.45, 0.10),
                    (1.00, 1.00, 0.97, 0.72))

function heat_color(value::Float64)
    v = clamp(value, 0.0, 1.0)
    for k in 1:(length(HEAT_STOPS) - 1)
        lo = HEAT_STOPS[k]
        hi = HEAT_STOPS[k + 1]
        if v <= hi[1]
            f = (v - lo[1]) / (hi[1] - lo[1])
            return (lo[2] + f * (hi[2] - lo[2]),
                    lo[3] + f * (hi[3] - lo[3]),
                    lo[4] + f * (hi[4] - lo[4]))
        end
    end
    top = HEAT_STOPS[end]
    return (top[2], top[3], top[4])
end

"""
    paint_relevance_map!(drawing, prot, state; grid=48, floor=1e-4)

Fill the pane with `R` sampled on a `grid x grid` lattice. `R` is displayed in
absolute units (`sum(weights) == 1`), so brightness is comparable across frames
and across trials.
"""
function paint_relevance_map!(drawing,
                              prot::GDFixation,
                              state::GDFixationState;
                              grid::Int = 48,
                              floor::Float64 = 1e-4)
    isempty(state.targets) && return nothing

    x0 = Float64(prot.bounds[1][1])
    y0 = Float64(prot.bounds[1][2])
    x1 = Float64(prot.bounds[2][1])
    y1 = Float64(prot.bounds[2][2])

    sigma = Float64(prot.sigma_fovea)
    table = relevance_grid(state.targets, state.weights, sigma, (x0, y0, x1, y1), grid)
    cw = (x1 - x0) / grid
    ch = (y1 - y0) / grid

    for j in 1:grid
        for i in 1:grid
            value = table[i, j]
            if value < floor
                continue
            end
            r, g, b = heat_color(value)
            cx = x0 + (i - 0.5) * cw
            cy = y0 + (j - 0.5) * ch
            @layer begin
                setopacity(1.0)
                sethue(r, g, b)
                # Slight overlap hides seams between cells.
                Luxor.box(Luxor.Point(cx, -cy), cw + 0.75, ch + 0.75, :fill)
            end
        end
    end
    return nothing
end

"""
    paint_fixation_overlay!(drawing, prot, state)

Draw the current fixation (white cross), the proposal (magenta circle), the
task-relevance support points (grey, radius proportional to weight) and a text
block with `R` at both positions plus the gain and the saccade threshold.
"""
function paint_fixation_overlay!(drawing,
                                 prot::GDFixation,
                                 state::GDFixationState)
    isempty(state.targets) && return nothing

    sigma = Float64(prot.sigma_fovea)
    fix_x = Float64(state.fixation[1])
    fix_y = Float64(state.fixation[2])
    pro_x = Float64(state.proposal[1])
    pro_y = Float64(state.proposal[2])

    r_fix = relevance_density(state.targets, state.weights, sigma, fix_x, fix_y)
    r_pro = relevance_density(state.targets, state.weights, sigma, pro_x, pro_y)

    # Support points
    @layer begin
        setopacity(0.9)
        for i in eachindex(state.targets)
            x = state.targets[i][1]
            y = state.targets[i][2]
            radius = 2.0 + 10.0 * state.weights[i]
            sethue(0.75, 0.75, 0.75)
            setline(1.0)
            Luxor.circle(Luxor.Point(x, -y), radius, :stroke)
        end
    end

    # Proposal
    @layer begin
        sethue(1.0, 0.2, 0.9)
        setline(1.6)
        Luxor.circle(Luxor.Point(pro_x, -pro_y), 9.0, :stroke)
    end

    # Current fixation: crosshair
    @layer begin
        sethue(1.0, 1.0, 1.0)
        setline(1.8)
        Luxor.line(Luxor.Point(fix_x - 9.0, -fix_y), Luxor.Point(fix_x + 9.0, -fix_y), :stroke)
        Luxor.line(Luxor.Point(fix_x, -fix_y - 9.0), Luxor.Point(fix_x, -fix_y + 9.0), :stroke)
        setline(1.0)
        Luxor.circle(Luxor.Point(fix_x, -fix_y), Float64(prot.sigma_fovea), :stroke)
    end

    # Readout
    half_w = drawing.width / 2.0
    half_h = drawing.height / 2.0
    @layer begin
        sethue(1.0, 1.0, 1.0)
        fontsize(10)
        text(@sprintf("R(fix)  %.4f", r_fix),
             Luxor.Point(-half_w + 8.0, -half_h + 14.0), halign = :left, valign = :top)
        text(@sprintf("R(prop) %.4f", r_pro),
             Luxor.Point(-half_w + 8.0, -half_h + 27.0), halign = :left, valign = :top)
        text(@sprintf("gain %.4f  eta %.4f", state.gain, prot.eta_saccade),
             Luxor.Point(-half_w + 8.0, -half_h + 40.0), halign = :left, valign = :top)
        text(@sprintf("sigma %.0f", sigma),
             Luxor.Point(-half_w + 8.0, -half_h + 53.0), halign = :left, valign = :top)
    end
    return nothing
end

"""
    paint_state(f::MentalModule{<:GDFixation}, ret_finish=true; kwargs...)

Pane showing the spatial task-relevance field the fixation controller optimises.
White cross: current fixation. Magenta ring: the fixation the optimiser would
commit to. Grey rings: relevance support points, radius proportional to mass.
"""
function paint_state(f::MentalModule{<:GDFixation},
                     ret_finish::Bool = true;
                     canvas_dims::Tuple{Float64, Float64} = (400.0, 400.0),
                     back_color::String = "black",
                     grid::Int = 48)
    prot, state = mparse(f)
    drawing = init_viz!(canvas_dims[1], canvas_dims[2], back_color)
    paint_relevance_map!(drawing, prot, state; grid = grid)
    paint_fixation_overlay!(drawing, prot, state)
    ret_finish && finish()
    return drawing
end
