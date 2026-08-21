export PFPerception, PFPerceptionState

struct PFPerception <: PerceptionProtocol
    pf::PFProtocol
    initial_args::Tuple
    initial_constraints::ChoiceMap
end

state_type(::Type{PFPerception}) = PFPerceptionState

# State definition
mutable struct PFPerceptionState <: MentalState{PFPerception}
    chain::PFChain
end

# State constructor
PFPerceptionState(v::PFPerception) =
    PFPerceptionState(
        PFChain(v.pf, s_model, v.initial_args, v.initial_constraints)
    )

function MentalModule(p::PFPerception)
    MentalModule(p, PFPerceptionState(p))
end

const WM_ARG_DIFFS =
    (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())

function step_args(p::PFPerception, t::Int)
    (_, istate, wm) = p.initial_args
    args = (t, istate, wm)
    (args, WM_ARG_DIFFS)
end

function step_module!(perception::MentalModule{V},
                      t::Int,
                      obs::ChoiceMap,
                      ) where {V<:PFPerception}
    p, q = mparse(perception)
    # Read the actual starting state and world model from the current particle trace
    first_trace = q.chain.particles.traces[1]
    _, curr_istate, curr_wm = get_args(first_trace)
    
    args = (t, curr_istate, curr_wm)
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    # Preattentive step
    inference_step!(q.chain, p.pf, args, argdiffs, obs)
    return nothing
end

function paint_state(m::MentalModule{PFPerception}, ret_finish=true)
    # unpack data
    p, q = mparse(m)
    _, istate, wm = p.initial_args
    particles = q.chain.particles.traces # REVIEW: this are not `unweighted`

    # initialize
    x, y = wm.motion.dimensions
    drawing = init_viz!(x, y)


    # draw particle states
    opacity = 3.0 / length(particles)
    for particle = particles
        ws = get_last_state(particle)
        for x = ws.objects
            paint!(x, S3V(1,1,1), opacity)
        end
    end
	ret_finish && finish()
    return drawing
end

"""
    reinit_perception!(perception::MentalModule{PFPerception}, new_fixation::SVector{2, Float32})

Re-initializes the particle filter traces in the perception module with a new fixation coordinate.
Each particle continues from its last inferred world state (`get_last_state`), while time is reset to 0.
"""
function reinit_perception!(perception::MentalModule{PFPerception}, new_fixation::SVector{2, Float32})
    p, q = mparse(perception)
    (_, istate_orig, wm_orig) = p.initial_args
    
    # 1. updated graphics model with new fixation point
    wm_orig.graphics.fixation_buf[1] = new_fixation[1]
    wm_orig.graphics.fixation_buf[2] = new_fixation[2]
    new_wm = wm_orig

    # 2. Extract last state from each particle and construct new seed traces
    traces = q.chain.particles.traces
    n_particles = length(traces)
    
    for i in 1:n_particles
        last_ws = get_last_state(traces[i])
        args = (0, last_ws, new_wm)
        q.chain.particles.traces[i], _ =
            generate(s_model, args, choicemap())
    end
    
    # 3. Update particle filter chain state
    # q.chain.particles.traces = new_traces
    # q.chain.particles.log_weights = zeros(Float64, n_particles)
    fill!(q.chain.particles.log_weights, 0.0)
    
    return nothing
end
