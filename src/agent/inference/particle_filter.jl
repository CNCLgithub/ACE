export PFProtocol,
    PFChain


"""
    ($TYPEDEF)

Perception Protocol that implements the adaptive computation interface in a particle filter.

---

$(TYPEDFIELDS)

"""
@kwdef struct PFProtocol <: InferenceProtocol
    "Number of particles"
    particles::Int = 1
    "Effective sample size"
    ess::Real = particles * 0.5
end

# REVIEW: could flatten data structure?
mutable struct PFChain
    particles::Gen.ParticleFilterState
end

function PFChain(p::PFProtocol, model, args::Tuple,
                 constraints::ChoiceMap)
    particles =
        Gen.initialize_particle_filter(model,
                                       args,
                                       constraints,
                                       p.particles)
    PFChain(particles)
end


"""

$(SIGNATURES)

Internally samples particles for the next time step, conditioning on observations.
"""
function inference_step!(chain::PFChain,
                         proc::PFProtocol,
                         args::Tuple,
                         argdiffs::Tuple,
                         obs::ChoiceMap)
    @show first(args)
    # update the state of the particles
    Gen.particle_filter_step!(chain.particles, args, argdiffs, obs)
    # Resample before moving on...
    maybe_resample!(chain.particles)
    return nothing
end

# function estimate_marginal(chain::PFChain{<:IncrementalQuery, <:PFProtocol},
#                            func::Function,
#                            args::Tuple)
#     @unpack state = chain
#     ws = state.log_weights
#     mass = logsumexp(ws)
#     acc = -Inf
#     @inbounds for i = 1:length(ws)
#         v = func(args..., state.traces[i])
#         w = ws[i] - mass
#         acc = logsumexp(acc, w + v)
#     end
#     if isnan(acc)
#         msg = "Marginal of $(func) lead to NaN, ws = $(ws);"
#         foreach(find_inf_scores, state.traces)
#         error(msg)
#     end
#     return acc
# end

# """
# Returns the MAP trace
# """
# function retrieve_map(chain::APChain)
#     @unpack state = chain
#     state.traces[argmax(state.log_weights)]
# end


# function reinit_chain(chain::APChain, template::STrace,
#                       cm = choicemap())
#     pf = estimator(chain)
#     q = estimand(chain)
#     steps = chain.steps # dt
#     _, wm, _ = q.args
#     ws = get_last_state(template)
#     args = (0, wm, ws)
#     q = IncrementalQuery(q.model, cm, args, INERTIA_ARG_DIFFS, 1)
#     Gen_Compose.initialize_chain(pf, q, steps)
# end
