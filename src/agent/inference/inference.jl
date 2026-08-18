export InferenceProtocol

"An algorithmic implementation of an inference procedure"
abstract type InferenceProtocol end

include("proposals.jl")
include("particle_filter.jl")
