export TargetDesignation,
    RGCollisionState,
    decision_expectation

"""
    ($TYPEDEF)

Protocol for estimating the marginal target probabilities so far.

---

$(TYPEDFIELDS)

"""
@kwdef struct TargetDesignation <: PlanningProtocol
    ntarget::Int64 = 4
end

mutable struct TDState <: MentalState{TargetDesignation}
    chain::Vector{Trace}
    expectation::Float64
    uncertainty::Float64
end

function TDState(p::TargetDesignation)
    TDState(Trace[], 0.5, 1.0)
end

function MentalModule(p::TargetDesignation)
    MentalModule(p, TDState(p))
end

# helper to extract planning state
function decision_expectation(pm::MentalModule{T}) where {T<:TargetDesignation}
    _, state = mparse(pm)
    state.expectation
end

function select_targets(ws::WorldState, ntargets::Int64)
    n = length(ws.objects)
    n > ntargets || error("Too many specified targets")

    elems = Vector{RFE_S2V}(undef, ntargets + 1)
    @inbounds for i = 1:ntargets
        obj = ws.objects[i]
        elems[i] =
            BernoulliElement{S2V}(0.95, normal_s2v, (obj.pos, 2.0))
    end

    ndistractors = n - ntargets
    positions = map(x -> x.pos, ws.objects[(ntargets+1:n)])
    mu = S2V(mean(positions))
    sigma = sum(std(positions; mean=mu))
    elems[ntargets + 1] = PoissonElement(
        Float64(ndistractors),
        normal_s2v,
        (mu, sigma)
    )
    return elems
end

include("gen.jl")

function select_cm(state::WorldState)
    n = length(state.objects)
    choicemap(((:xs => i, state.objects[i].pos) for i = 1:n)...)
end

function seed_model(p::TargetDesignation, tr::STrace)
    istate = get_last_state(tr)
    args = (istate, p.ntarget)
    cm = select_cm(istate)
    trace, _ =
        generate(target_designation, args, cm)
    return trace
end

function seed_state!(state::TDState, protocol::TargetDesignation,
                     perception::MentalModule{PFPerception})

    vp, vs = mparse(perception)
    traces = Vector{Trace}(undef, vp.pf.particles)

    traces = sample_unweighted_traces(vs.chain.particles, vp.pf.particles)
    # vtraces = vs.chain.particles.traces
    # @inbounds for i=1:vp.pf.particles
        # st = get_last_state(vtraces[i])
    #     traces[i] = seed_model(protocol, traces[i])
    # end
    state.chain = traces
    return nothing
end

"""

$(SIGNATURES)

Computes the marginal over collision counts.

Also updates the \$\\delta \\pi\$ records in the attention module.
"""
function step_module!(planner::MentalModule{T},
                      t::Int,
                      perception::MentalModule{V}
                      ) where {T<:TargetDesignation,
                               V<:PerceptionProtocol}
    # Propogate info from perception
    protocol, state = mparse(planner)
    seed_state!(state, protocol, perception)
    # Initial estimate of marginal red-green ratio
    update_expectation!(state, protocol)
    return nothing
end

function update_expectation!(m::MentalModule{TargetDesignation})
    p, state = mparse(m)
    update_expectation!(state, p)
    return nothing
end

function update_expectation!(state::TDState, p::TargetDesignation)
    new_expectation =
        approximate_td_marginal(p, state.chain)
    # smoothing
    state.expectation = 0.1*state.expectation + 0.9*new_expectation
    return nothing
end

function approximate_td_marginal(p::TargetDesignation, traces::Vector)
    # REVIEW: weighted mean?
    # exp_pi = mean(expected_reward_td_rfs, traces)
    exp_pi = mean(tr -> expected_reward_td_conf(p, tr), traces)
end

function expected_reward_td_conf(p::TargetDesignation, trace::STrace)
    state = get_last_state(trace)
    expected_reward_td_conf(p, state)
end

function expected_reward_td_conf(p::TargetDesignation, state::WorldState)
    n = length(state.objects)
    n > p.ntarget || error("Not enough objects for target designation")

    min_d = Inf
    @inbounds for i = 1:p.ntarget
        target = state.objects[i]
        for j = (p.ntarget+1):n
            distractor = state.objects[j]
            d = norm(target.pos - distractor.pos)
            min_d = min(min_d, d)
        end
    end

    # reward = exp(-total_d)
    log_reward = min_d
end

function expected_reward_td_rfs(trace::PiTrace, temp::Float64 = 1.0)
    (_, ntargets) = get_args(trace)
    rfs = extract_rfs_subtrace(trace)
    pt = rfs.ptensor
    nx,ne,np = size(pt)

    # normalize the log scores of each partition
    nls = log.(softmax(rfs.pscores, temp))
    
    # probability that each observation
    # is explained by a target
    x_weights = Vector{Float64}(undef, nx)
    for x = 1:nx
        xw = -Inf
        for p = 1:np
            if !pt[x, ne, p]
                xw = logsumexp(xw, nls[p])
            end
        end
        x_weights[x] = xw
    end

    # the ratio of observations explained by each target
    # weighted by the probability that the observation is
    # explained by other targets
    td_weights = fill(-Inf, ntargets)
    for i = 1:ntargets
        for p = 1:np
            ew = -Inf
            for x = 1:nx
                pt[x, i, p] || continue
                ew = x_weights[x]
                # assuming isomorphicity
                # (one association per partition)
                break
            end
            # P(e -> x) where x is associated with any other targets
            ew += nls[p]
            td_weights[i] = logsumexp(td_weights[i], ew)
        end
    end
    logsumexp(td_weights) - log(ntargets)
end


function proxy_delta_pi(m::MentalModule{TargetDesignation}, tr::STrace, i::Int)
    protocol, dm_state = mparse(m)
    pi = expected_reward_td_conf(protocol, dm_state.chain[i])
    new_pi = expected_reward_td_conf(protocol, tr)
    # delta_pi = abs(new_pi - pi)
    delta_pi = logabsdiffexp(new_pi, pi)
    tr, delta_pi
end

function update_pi_trace(tr::PiTrace, new_state::WorldState)
    # (orig_state, ntargets) = get_args(tr)

    # args = (orig_state, ntargets)
    # argdiffs = (NoChange(), NoChange())
    # cm = select_cm(new_state)
    # new_tr, w, _... = update(tr, args, argdiffs, cm)
    # delta_pi = abs(w)
    # new_tr, _... = generate(get_gen_fn(tr), args)
    # pi = expected_reward_td_rfs(tr, 10.0)
    # new_pi = expected_reward_td_rfs(new_tr, 10.0)
    # delta_pi = log(abs(new_pi - pi))
    # @show pi
    # @show new_pi
    # @show delta_pi
    # error()
    return new_tr, delta_pi
end

function update_planning!(m::MentalModule{TargetDesignation}, new_trace, i::Int)
    _, state = mparse(m)
    state.chain[i] = new_trace
    return nothing
end
