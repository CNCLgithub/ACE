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
    args, argdiffs = step_args(p, t)
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
