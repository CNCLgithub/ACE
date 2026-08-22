export FixationProtocol, GDFixation

@kwdef struct GDFixation <: FixationProtocol
    eta_saccade::Float64 = 0.05
    lr::Float64 = 1.0
    momentum::Float64 = 0.9
    num_steps::Int64 = 100
    bounds::Tuple{S2V, S2V} = (S2V(-400.0, -400.0), S2V(400.0, 400.0))
    tau_importance::Float64 = 1.0
    sigma_fovea::Float64 = 50.0
    gamma::Float64 = 0.9
    lambda_l2::Float64 = 0.0001
    lambda_smooth::Float64 = 0.0005
end

mutable struct GDFixationState <: MentalState{GDFixation}
    fixation::MVector{2, Float32}
    fixation_vel::MVector{2, Float32}
end

GDFixationState() = GDFixationState(MVector{2, Float32}(0.0f0, 0.0f0), MVector{2, Float32}(0.0f0, 0.0f0))
MentalModule(p::GDFixation) = MentalModule(p, GDFixationState())

function step_module!(f::MentalModule{F}, t::Int64, v::MentalModule{V}, a::MentalModule{A}
                      ) where {F<:GDFixation, V<:PFPerception, A<:AdaptiveComputation}
    # Check if attention map is ready
    isready(a) || return nothing
    amap, weights = attention_map(a, v)

    # 1. Gradient descent on gaze coordinates
    fprot, fstate = mparse(f)
    opt_fix!(fstate, fprot, amap, weights)
    return nothing
end


"""
    foveal_loss_and_grad(fixation, targets, weights, sigma_fovea)

Computes loss and analytical gradient for weighted Gaussian coverage:
L(f) = -Σ w_i * exp(-||x_i - f||^2 / (2σ^2))
∇L(f) = -Σ w_i * exp(-||x_i - f||^2 / (2σ^2)) * ((x_i - f) / σ^2)
"""
@inline function foveal_loss_and_grad!(grad::MVector{2, Float32},
                                       fixation::SVector{2, Float32},
                                       targets::AbstractVector{S2V},
                                       weights::AbstractVector{Float64},
                                       sigma_fovea::Float32)
    n = length(targets)
    inv_2sigma_sq = 1.0f0 / (2.0f0 * sigma_fovea * sigma_fovea)
    inv_sigma_sq  = 1.0f0 / (sigma_fovea * sigma_fovea)

    loss = 0.0f0
    fill!(grad, 0.0f0)

    @inbounds for i in 1:n
        dx = Float32(targets[i][1]) - fixation[1]
        dy = Float32(targets[i][2]) - fixation[2]
        d_sq = dx * dx + dy * dy + 1E-3
        w = Float32(weights[i])

        cov = exp(-d_sq * inv_2sigma_sq)
        loss -= w * cov

        if isinf(loss)
            @show d_sq
            @show w
            @show cov
            error("Inf loss encountered")
        end

        # ∇L = -w * cov * (target - fix) / σ²
        g_factor = -w * cov * inv_sigma_sq
        inc = SVector{2, Float32}(g_factor * dx, g_factor * dy)
        # if iszero(inc)
        #     @show -w
        #     @show cov
        #     @show inv_sigma_sq
        # end
        grad .+= inc
    end
    # @show grad
    # error()
    return loss
end

"""
    opt_fix!(fstate, fprot, amap, weights)

Performs 2D momentum gradient descent to resolve next fixation coordinates.
"""
function opt_fix!(fstate::GDFixationState,
                  fprot::GDFixation,
                  amap::AbstractVector{S2V},
                  weights::AbstractVector{Float64})
    n = length(amap)
    n == 0 && return nothing

    f_prev = SVector{2, Float32}(fstate.fixation)
    v_prev = SVector{2, Float32}(fstate.fixation_vel)
    f_curr = f_prev
    momentum_buf = MVector{2, Float32}(0.0f0, 0.0f0)
    cov_grad = MVector{2, Float32}(0.0f0, 0.0f0)

    sigma_f = Float32(fprot.sigma_fovea)
    lr = Float32(fprot.lr)
    mom = Float32(fprot.momentum)
    l2 = Float32(fprot.lambda_l2)
    smooth = Float32(fprot.lambda_smooth)

    min_b = SVector{2, Float32}(fprot.bounds[1][1], fprot.bounds[1][2])
    max_b = SVector{2, Float32}(fprot.bounds[2][1], fprot.bounds[2][2])

    # Initial loss at f_t
    init_cov_loss = foveal_loss_and_grad!(cov_grad, f_prev, amap, weights, sigma_f)
    init_loss = init_cov_loss
    final_loss = init_loss

    # 100-step Momentum GD Loop (Stack-only, 0 allocations)
    for _ in 1:fprot.num_steps
        final_loss = foveal_loss_and_grad!(cov_grad, f_curr, amap, weights, sigma_f)
        
        # Movement cost gradients:
        # L_l2 = 0.5 * l2 * ||f - f_prev||^2         => ∇ = l2 * (f - f_prev)
        # L_smooth = 0.5 * smooth * ||f - f_prev - v_prev||^2 => ∇ = smooth * (f - f_prev - v_prev)
        disp = f_curr - f_prev
        move_grad = l2 * disp + smooth * (disp - v_prev)

        # @show cov_grad
        total_grad = cov_grad + move_grad

        momentum_buf = mom * momentum_buf + lr * total_grad
        f_curr = clamp.(f_curr - momentum_buf, min_b, max_b)
    end

    # Evaluate saccade gain threshold
    gain = init_loss - final_loss
    # @show init_loss
    # @show final_loss
    # @show gain

    if gain > Float32(fprot.eta_saccade)
        # Commit saccade
        fstate.fixation_vel[1] = f_curr[1] - fstate.fixation[1]
        fstate.fixation_vel[2] = f_curr[2] - fstate.fixation[2]
        fstate.fixation[1] = f_curr[1]
        fstate.fixation[2] = f_curr[2]
    else
        # Fixation holds
        fstate.fixation_vel .= 0.0f0
    end

    return nothing
end
