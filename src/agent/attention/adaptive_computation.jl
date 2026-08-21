export AdaptiveComputation

################################################################################
# Adaptive Computation
################################################################################

@kwdef struct AdaptiveComputation{V, C} <: AttentionProtocol
    vis_partition::TracePartition{V} = WMPartition{V}()
    cog_partition::TracePartition{C} = WMPartition{C}()
    "Minimum number of moves"
    base_steps::Int64 = 3
    "Size of attention hash map"
    buffer_size::Int64 = 100
    "Distance metric in spatio-temporal maps (x,y,t)"
    map_metric_weights::S3V = S3V(1/3, 1/3, 1/3)
    map_metric::PreMetric = WeightedEuclidean(map_metric_weights)
    "Number of nearest neighbors"
    nns::Int64 = 5
    "Importance softmax temperature"
    itemp::Float64 = 3.0
    "Maximumal load"
    load::Int64 = 20
    "Load curve slope"
    load_m::Float64 = 20.0
    "Load curve intercept"
    load_x0::Float64 = 5.0
end

mutable struct AdaptiveAux <: MentalState{AdaptiveComputation}
    "Impact of C_k on decision-making"
    dPi::HashMap
    "Impact of C_k on thinking"
    dK::HashMap
    "Impact of C_k on perception"
    dS::HashMap
    "Array used for task-relevance integration indeces"
    nn_idxs::Vector{Int32}
    "Array used for task-relevance integration distances"
    nn_dists::Vector{Float64}
    "Statistic over average load"
    avg_load::Float64
end

AdaptiveAux(n::Int, k::Int) = AdaptiveAux(HashMap(S3V, Float64,   n), # dPi
                                          HashMap(S3V, Float64,   n), # dK
                                          HashMap(S3V, Float64,   n), # dS
                                          zeros(Int32, k),
                                          zeros(Float64, k),
                                          0.0
                                          ) 

function MentalModule(m::AdaptiveComputation)
    MentalModule(m, AdaptiveAux(m.buffer_size, m.nns))
end

function isready(m::MentalModule{<:AdaptiveComputation})
    _, state = mparse(m)
    !isempty(state)
end

Base.isempty(x::AdaptiveAux) = isempty(x.dPi)
# Base.isempty(x::AdaptiveAux) = isempty(x.dPi) || isempty(x.dS) || isempty(x.dK)

function update_impact!(buffer::HashMap,
                        partition::TracePartition{T},
                        trace::T,
                        j::Int,
                        delta::Float64,
                        ) where {T<:Trace}
    coord = get_coord(partition, trace, j)
    push_sample!(buffer, coord, delta)
    return nothing
end

function update_task_relevance!(att::MentalModule{A}
                                ) where {A<:AdaptiveComputation}
    attp, attstate = mparse(att)
    fit_map!(attstate.dPi, attp.map_metric)
    fit_map!(attstate.dK , attp.map_metric)
    fit_map!(attstate.dS , attp.map_metric)
    return nothing
end

function load(p::AdaptiveComputation, deltas::Vector{Float64})
    x = logsumexp(deltas)
    x = (x - p.load_x0) / p.load_m
    l = p.load * exp(min(x, 0.0))
    # println("| Agg. Delta: $(round(x; digits=2)) | Load: $(l)")
    return l
end

function task_relevance!(aux::AdaptiveAux,
                         dPi::HashMap,
                         dS::HashMap,
                         partition::TracePartition{T},
                         trace::T,
                         ) where {T<:Gen.Trace}
    n = latent_size(partition, trace)
    # NOTE: case with empty estimate?
    # No info yet -> -Inf
    (isempty(dPi) || isempty(dS)) && return fill(-Inf, n)
    tr = Vector{Float64}(undef, n)
    # Preallocating reused arrays
    for i = 1:n
        coord = get_coord(partition, trace, i)
        _dpi  = integrate!(aux.nn_idxs, aux.nn_dists, coord, dPi)
        # _ds   = integrate!(aux.nn_idxs, aux.nn_dists, coord, dS)
        # @printf "| δπ: %.2f \t | δS: %.2f |\n" _dpi _ds
        # @printf "| δπ: %.2f \t |\n" _dpi
        tr[i] = _dpi #+ _ds
    end
    return tr
end

function step_module!(att::MentalModule{AdaptiveComputation},
                      t::Int,
                      vis::MentalModule{PFPerception},
                      planning::MentalModule{TargetDesignation})
    update_task_relevance!(att)
    attend!(att, vis, planning)
    update_expectation!(planning)
    # attend!(att, planning)
    return nothing
end

function attend!(att::MentalModule{AdaptiveComputation},
                 perception::MentalModule{PFPerception},
                 planning::MentalModule{TargetDesignation})

    _, vstate = mparse(perception)
    protocol, aux = mparse(att)

    @unpack vis_partition, base_steps, itemp = protocol
    pf_state = vstate.chain.particles

    np = length(pf_state.traces)

    avg_load = 0.0
    avg_mag_delta = -Inf

    for i = 1:np # iterate through each particle
        trace = pf_state.traces[i]
        # determine the importance of each latent
        deltas = task_relevance!(aux, aux.dPi, aux.dS, vis_partition, trace)
        avg_mag_delta = logsumexp(avg_mag_delta, logsumexp(deltas))
        importance = softmax(deltas, itemp)
        tload = load(protocol, deltas)
        avg_load += tload
        nobj = length(deltas)
        steps_per_obj = floor(Int, base_steps / nobj)
        # Stage 2
        # select latent and C_k
        for j = 1:nobj
            steps = steps_per_obj + round(Int, tload * importance[j])
            for _ = 1:steps
                prop = select_prop(vis_partition, trace, j)
                # Apply computation
                new_trace, alpha = prop(trace, j)

                # Estimate and update impacts
                # delta S
                dS = min(alpha, 0.)
                update_impact!(aux.dS, vis_partition, trace, j, dS)
                # delta pi
                new_pi, dPi = proxy_delta_pi(planning, new_trace, i)
                update_impact!(aux.dPi, vis_partition, trace, j, dPi)
                # if j == 2 && dPi > 0.5
                #     error()
                # end
                # Update particle
                if log(rand()) < alpha
                    trace = new_trace
                    update_planning!(planning, new_pi, i)
                    # pi = new_pi
                    pf_state.log_weights[i] += alpha
                end
            end
        end
        pf_state.traces[i] = trace
    end

    avg_mag_delta = avg_mag_delta - log(np)
    # @show aux.avg_load
    # Time smoothing
    aux.avg_load = avg_load / np
    # aux.avg_load = 0.2 * aux.avg_load + 0.8 * avg_load / np
    # @printf "|Δ|: %5.2f, Avg. Load: %5.2f\n" avg_mag_delta aux.avg_load
    return nothing
end

function attend!(att::MentalModule{<:AdaptiveComputation},
                 planning::MentalModule{<:TargetDesignation})


    protocol, aux = mparse(att)
    @unpack cog_partition, base_steps, itemp = protocol

    plan_prot,  state = mparse(planning)
    n = length(state.chain)

    for i = 1:n # iterate through each particle
        trace = state.chain[i]
        # determine the importance of each latent
        deltas = task_relevance!(aux, aux.dPi, aux.dK, cog_partition, trace)
        importance = softmax(deltas, itemp)
        tload = load(protocol, deltas)
        nobj = length(deltas)
        steps_per_obj = floor(Int, base_steps / nobj)
        # Stage 2
        # select latent and C_k
        for j = 1:nobj
            steps = steps_per_obj + round(Int, tload * importance[j])
            for _ = 1:steps
                prop = select_prop(cog_partition, trace, j)
                # Apply computation, estimate dS
                new_trace, alpha, dPi = prop(trace, j)
                dK = min(alpha, 0.)
                if log(rand()) < alpha # update particle
                    trace = new_trace
                end
                update_impact!(aux.dK, cog_partition, trace, j, dK)
                update_impact!(aux.dPi, cog_partition, trace, j, dPi)
            end
        end

        state.chain[i] = trace
    end

    update_expectation!(state, plan_prot)

    return nothing
end

function attention_map(m::MentalModule{<:AdaptiveComputation})
    prot, state = mparse(m)

    dPi = state.dPi
    # REVIEW: any way to prevent new allocations?
    n = length(dPi.coords)
    amap = Vector{S2V}(undef, n)
    weights = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        x, y, _ = dPi.coords[i]
        amap[i] = S2V(x, y)
        weights[i] = dPi.samples[i]
    end
    (amap, weights)
end 
