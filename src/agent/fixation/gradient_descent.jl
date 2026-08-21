export FixationProtocol, GDFixation

@kwdef struct GDFixation <: FixationProtocol
    eta_saccade::Float64 = 0.05
    lr::Float64 = 200.0
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
    fixation::MVector{2, Float64}
    fixation_vel::MVector{2, Float64}

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
    # No attention scores yet
    @show isready(a)
    isready(a) || return nothing
    # 1. Attention map
    amap, weights = attention_map(a)
    # 2. Optimization
    fprot, fstate = mparse(f)
    opt_fix!(fstate, fprot, amap, weights)
    return nothing
end

function opt_fix!(fstate::GDFixationState, fprot::GDFixation,
                  amap::Array{S2V}, weights::Array{Float64})
    outputs_py = jax_script[].optimize_fixation(
        fstate.fixation_buf_py,
        fstate.fixvel_buf_py,
        Py(amap),
        Py(weights)
    )
    fstate.fixvel_buf_py = outputs_py - fstate.fixation_buf_py
    fstate.fixation_buf_py = outputs_py
    @show outputs_py
    return nothing
end
