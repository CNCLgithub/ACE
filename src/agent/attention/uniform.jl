export UniformProtocol

################################################################################
# Uniform Rationing
################################################################################

@with_kw struct UniformProtocol <: AttentionProtocol
    "Number of rejuvenation moves"
    moves::Int64 = 10
    partition::TracePartition = WMPartition()
end

struct UniformAuxState <: MentalState{UniformProtocol} end

function AuxState(::UniformProtocol)
    UniformAuxState()
end

function step_module!(att::MentalModule{<:UniformProtocol},
                      t::Int,
                      vis::MentalModule{<:PFProtocol})
    visp, visstate = mparse(vis)
    attend!(visstate, visp, att)
    return nothing
end

function attend!(chain::PFChain,
                 pf::PFProtocol,
                 att::MentalModule{<:UniformProtocol})

    protocol, aux = mparse(att)
    state = chain.particles

    np = length(state.traces)

    for i = 1:np # iterate through each particle
        trace = state.traces[i]
        nobj = representation_count(trace)
        # number of moves per object
        steps_per_obj = round(Int, protocol.moves / nobj)
        # Stage 2
        # select latent and C_k
        for j = 1:nobj
            for _ = 1:steps_per_obj
                prop = get_prop(protocol.partition, trace, j)
                # Apply computation, estimate dS
                new_trace, alpha = prop(trace)
                if log(rand()) < alpha # update particle
                    trace = new_trace
                    state.log_weights[i] += alpha
                end
            end
        end

        state.traces[i] = trace
    end
    return nothing
end

function AttentionModule(m::UniformProtocol)
    MentalModule(m, UniformAuxState())
end

# HACK: dummy function - called in collision counter
function update_dPi!(att::MentalModule{A},
                     obj::WMObject,
                     delta::Float64) where {A<:UniformProtocol}
    return nothing
end

