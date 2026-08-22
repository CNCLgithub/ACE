export GraphicsModel, RFGraphics, ReceptiveFields, receptive_fields

include("receptive_field.jl")

mutable struct RFGraphics <: GraphicsModel
    geom::RFGeometry
    fixation::MVector{2, Float32}
    state::WorldState
    
    # Pre-allocated scratch buffers
    means_buf::Matrix{Float32}
    vars_buf::Matrix{Float32}

    mu_rgb::MVector{3, Float32}
    var_rgb::MVector{3, Float32}
end

function RFGraphics(image_shape::Tuple{Int, Int},
                    initial_fixation::SVector{2, Float32},
                    initial_scene::WorldState;
                    kwargs...)
    geom = generate_receptive_fields(image_shape; kwargs...)
    n_rf = length(geom.fields)
    means_buf = Matrix{Float32}(undef, n_rf, 3)
    vars_buf  = Matrix{Float32}(undef, n_rf, 3)

    mu_rgb = MVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
    var_rgb = MVector{3, Float32}(0.0f0, 0.0f0, 0.0f0)
    
    RFGraphics(geom, MVector{2, Float32}(initial_fixation), initial_scene, means_buf, vars_buf,
               mu_rgb, var_rgb)
end

function sync_scene(graphics::RFGraphics, scene::WorldState)
    graphics.state = scene
    return graphics
end
function sync_scene(graphics::RFGraphics, scene::WorldState, fixation::S2V)
    graphics.state = scene
    graphics.fixation[1] = Float32(fixation[1])
    graphics.fixation[2] = Float32(fixation[2])
    return graphics
end

function field_predict(graphics::RFGraphics)
    predict_rf_stats!(graphics.means_buf, graphics.vars_buf,
                      graphics.mu_rgb, graphics.var_rgb,
                      SVector(graphics.fixation), graphics.geom, graphics.state.objects)
    
    n_rf = size(graphics.means_buf, 1)
    sample = Matrix{Float32}(undef, n_rf, 3)
    @inbounds for c in 1:3, i in 1:n_rf
        sample[i, c] = graphics.means_buf[i, c] + sqrt(graphics.vars_buf[i, c]) * randn(Float32)
    end
    return sample
end

# ------------------------------------------------------------------------------
# Gen Distribution
# ------------------------------------------------------------------------------
struct ReceptiveFields <: Distribution{Matrix{Float32}} end
const receptive_fields = ReceptiveFields()

function Gen.random(::ReceptiveFields, graphics::RFGraphics)
    field_predict(graphics)
end

(::ReceptiveFields)(graphics::RFGraphics) = Gen.random(receptive_fields, graphics)

function Gen.logpdf(::ReceptiveFields, x::AbstractMatrix{Float32}, graphics::RFGraphics)
    predict_rf_stats!(graphics.means_buf, graphics.vars_buf,
                      graphics.mu_rgb, graphics.var_rgb,
                      SVector(graphics.fixation), graphics.geom, graphics.state.objects)
    return rf_logpdf(x, graphics.means_buf, graphics.vars_buf)
end

Gen.has_argument_grads(::ReceptiveFields) = (false,)
Gen.has_output_grad(::ReceptiveFields) = false
Gen.is_discrete(::ReceptiveFields) = false
