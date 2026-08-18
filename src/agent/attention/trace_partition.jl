export TracePartition,
    latent_size,
    get_coord,
    select_prop,
    WMPartition

""" A procedure to selectively index and process traces"""
abstract type TracePartition{T<:Gen.Trace} end

"""
    latent_size(p::TracePartition{T}, tr::T) where {T<:Gen.Trace}

Determine the number of latents exposed for selective processing.
"""
function latent_size end


"""
    get_coord(p::TracePartition{T}, tr::T, i::Int) where {T<:Gen.Trace}

Get the representational coordinate of latent `i` in the trace.
"""
function get_coord end


@with_kw struct WMPartition{T} <: TracePartition{T}
    "Factor for time dimension"
    time_factor::Float64 = 1.0
end

const MODEL_TRACE = Union{WMTrace, KModelTrace, KMDTrace}

function latent_size(::WMPartition{T}, tr::T) where {T<:WMTrace}
    state = get_last_state(tr)
    length(state.dynamic)
end

function latent_size(::WMPartition{T}, tr::T) where {T<:KMDTrace}
    t, state, _... = get_args(tr)
    length(state.dynamic)
end

# REVIEW: encode shape or color?
function get_coord(p::WMPartition{T}, tr::T, idx::Int
                   ) where {T<:WMTrace}
    t, _... = get_args(tr)
    state = get_last_state(tr)
    _, pos, _ = state.dynamic[idx]
    x, y = pos
    S3V(x, y, p.time_factor * t)
end
function get_coord(p::WMPartition{T}, tr::T, idx::Int
                   ) where {T<:KMDTrace}
    t, state, _... = get_args(tr)
    _, pos, _ = state.dynamic[idx]
    x, y = pos
    S3V(x, y, p.time_factor * t)
end

function get_time_coord(tr::WMTrace)
    t, _... = get_args(tr)
    return t
end

function get_time_coord(tr::KMDTrace)
    t, _... = get_args(tr)
    return t
end

function select_prop(::WMPartition, ::MODEL_TRACE, ::Int)
    object_ancestral_proposal
end
