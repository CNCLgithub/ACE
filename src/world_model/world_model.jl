export WorldModel, WorldState, Object, Disc


abstract type Object end

struct Disc <: Object
    pos::S2V
    vel::S2V
    radius::Float64
end

abstract type MotionModel end
abstract type GraphicsModel end

@kwdef struct WorldModel
    motion::MotionModel
    graphics::GraphicsModel
end

struct WorldState
    objects::Vector{Disc}
end

include("motion.jl")
include("graphics.jl")
include("gen.jl")
include("trace.jl")
include("visuals.jl")

