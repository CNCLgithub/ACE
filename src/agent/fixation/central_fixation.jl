export CentralFixation

struct CentralFixation <: FixationProtocol end

struct CFState <: MentalState{GDFixation} end

MentalModule(p::GDFixation) = MentalModule(p, CFState())

function step_module!(f::MentalModule{F}, t::Int64, v::MentalModule{V}, a::MentalModule{A}
                      ) where {F<:CentralFixation, V<:PFPerception, A<:AdaptiveComputation}
    # Check if attention map is ready
    isready(a) || return nothing
    amap, weights = attention_map(a, v)

    # 1. Gradient descent on gaze coordinates
    fprot, fstate = mparse(f)
    opt_fix!(fstate, fprot, amap, weights)
    return nothing
end

function get_fixation(::MentalModule{<:CentralFixation})
    return S2V(0, 0)
end
