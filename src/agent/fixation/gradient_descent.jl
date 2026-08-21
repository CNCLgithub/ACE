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


state_type(::Type{GDFixation}) = GDFixationState

# State definition
mutable struct GDFixationState <: MentalState{GDFixation}
    fixation::MVector{2, Float32}
    fixation_vel::MVector{2, Float32}

    # Python buffer protocol wrappers (Zero-copy views into Julia host arrays)
    fixation_buf_py::Py
    fixvel_buf_py::Py
end

function GDFixationState()
    # Allocate fixed host buffers in Julia
    fixation_buf = MVector{2, Float32}(0, 0)
    fixation_vel = MVector{2, Float32}(0, 0)

    # Create zero-copy Python views
    fixation_buf_py = Py(fixation_buf)
    fixvel_buf_py = Py(fixation_vel)

    GDFixationState(fixation_buf, fixation_vel, fixation_buf_py, fixvel_buf_py)
end

function MentalModule(p::GDFixation)
    MentalModule(p, GDFixationState())
end

function step_module!(f::MentalModule{F}, t::Int64, v::MentalModule{V}, a::MentalModule{A}
                      ) where {F<:GDFixation, V<:PFPerception, A<:AdaptiveComputation}
    # Check if attention map is ready
    isready(a) || return nothing
    amap, weights = attention_map(a)
    isempty(amap) && return nothing

    # 1. Gradient descent on gaze coordinates
    fprot, fstate = mparse(f)
    opt_fix!(fstate, fprot, amap, weights)
    return nothing
end

function opt_fix!(fstate::GDFixationState, fprot::GDFixation,
                  amap::AbstractVector{<:AbstractVector{<:Real}},
                  weights::AbstractVector{<:Real})
    n = length(amap)
    n == 0 && return nothing

    # Flatten coordinates & weights to Float32 host buffers
    flat_targets = Vector{Float32}(undef, 2 * n)
    flat_weights = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        flat_targets[2*i - 1] = Float32(amap[i][1])
        flat_targets[2*i]     = Float32(amap[i][2])
        flat_weights[i]       = Float32(weights[i])
    end

    outputs_py = jax_script[].optimize_fixation(
        fstate.fixation_buf_py,
        fstate.fixvel_buf_py,
        Py(flat_targets),
        Py(flat_weights),
        Int32(n),
        eta_saccade=fprot.eta_saccade,
        lr=fprot.lr,
        momentum=fprot.momentum,
        num_steps=fprot.num_steps,
        sigma_fovea=fprot.sigma_fovea,
        gamma=fprot.gamma,
        lambda_l2=fprot.lambda_l2,
        lambda_smooth=fprot.lambda_smooth
    )

    # Convert output back to Julia array and update fixation buffers
    new_fix = pyconvert(Vector{Float32}, outputs_py)

    fstate.fixation_vel[1] = new_fix[1] - fstate.fixation[1]
    fstate.fixation_vel[2] = new_fix[2] - fstate.fixation[2]

    fstate.fixation[1] = new_fix[1]
    fstate.fixation[2] = new_fix[2]

    return nothing
end
