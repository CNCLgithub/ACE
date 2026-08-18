export Agent, MentalProtocol,
    MentalState, MentalModule, mparse,
    PerceptionProtocol,
    PlanningProtocol,
    AttentionProtocol,
    step_agent!


"The algorithmic implementation of a mental process"
abstract type MentalProtocol end

"The state of a mental process"
abstract type MentalState{T<:MentalProtocol} end

"""
$(TYPEDEF)

An algorithmic "organ" - both its function (protocol) and form (state).

---

Each protocol should call this constructor. See [`PerceptionModule`](@ref)

"""
mutable struct MentalModule{T<:MentalProtocol}
    protocol::T
    state::MentalState{T}
end

# function MentalModule(protocol::T) where {T<:MentalProtocol}
#     S = state_type(T)
#     MentalModule{T}(protocol, S(protocol))
# end

"""
    $(TYPEDSIGNATURES)

Returns the protocol and state of a mental module.
"""
function mparse(m::MentalModule)
    (m.protocol, m.state)
end


"""
    $(FUNCTIONNAME)(module, t, ...)

Run the mental module forward for one tick.

Here, `t::Int` denotes the global clock step.
Not all modules will perform operations every tick.
Should return `Nothing`.

---

$(METHODLIST)
"""
function step_module! end

"Converts sensory signals to representations of the world"
abstract type PerceptionProtocol <: MentalProtocol end
"Digests perceptual inferences to determine rewarding actions"
abstract type PlanningProtocol <: MentalProtocol end
"Rations resources within inference procedures"
abstract type AttentionProtocol <: MentalProtocol end


"""
    $(TYPEDEF)

A simulated agent.

---

$(TYPEDFIELDS)

"""
mutable struct Agent{
                     V<:PerceptionProtocol,
                     P<:PlanningProtocol,
                     A<:AttentionProtocol
                    }
    "Observations -> Worlds"
    perception::MentalModule{V}
    "Worlds -> Goals"
    planning::MentalModule{P}
    "What to attend to in the world"
    attention::MentalModule{A}
end


"""
    (TYPEDSIGNATURES)

Simulate one tick in the agent's mind.

Not all modules will necessarily operate at each tick.
"""
function step_agent!(agent::Agent, t::Int, obs::ChoiceMap)
    @unpack attention, perception, planning = agent
    # advance state-tracking particle filter
    step_module!(perception, t, obs)
    # approximate red-green marginal over future states
    step_module!(planning, t, perception)
    # further refine current world state and future predictions
    step_module!(attention,  t, perception, planning)
    return nothing
end

# Mental module implementations
include("inference/inference.jl")
include("perception/perception.jl") 
include("planning/planning.jl") 
include("attention/attention.jl")

# agent-tailored visualizations
# include("visuals.jl")
# include("io.jl")
