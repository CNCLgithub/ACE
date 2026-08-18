@kwdef struct WorldModel
    motion::MotionModel
    graphics::GraphicsModel
end

abstract type Object end

struct Disc <: Object
    pos::S2V
    vel::S2V
    radius::Float64
end

abstract type MotionModel end
include("motion.jl")

abstract type GraphicsModel end
include("graphics.jl")
