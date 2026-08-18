# """

#     PlanningModule(::T, ...)::MentalModule{T} where {T<:PlanningProtocol}

# Constructor that should be implemented by each `PlanningProtocol`

# ---

# # Implementations:

# (METHODLIST)
# """
# function PlanningModule end

# function PlanningModule(planning::PlanningProtocol,
#                         inference::PFProtocol,
#                         query::SequentialQuery)
#     MentalModule(planning, PFChain(inference, query))
# end
include("collision.jl")
include("visuals.jl")
