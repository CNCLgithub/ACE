export GraphicsModel, RFGraphics, ReceptiveFields

mutable struct RFGraphics <: GraphicsModel
    # Python handle for receptive field geometry (stays on GPU)
    fields_py::Py

    # Pre-allocated Julia host buffers
    fixation_buf::MVector{2, Float32}
    scene_buf::Vector{SVector{7, Float32}}
    
    # Python buffer protocol wrappers (Zero-copy views into Julia host arrays)
    fixation_buf_py::Py
    scene_buf_py::Py
end

function scene_to_array(state::WorldState)
    n = length(state.objects)
    scene_buf = Vector{SVector{7, Float32}}(undef, n)
    @inbounds for i = 1:n
        obj = state.objects[i]
        x, y = obj.pos
        scene_buf[i] = [x, y, # position
                        obj.radius, # radius
                        0,          # shape - circle
                        1, 1, 1]    # rgb
    end
    return scene_buf
end

function RFGraphics(image_shape::Tuple{Int, Int},
                    initial_fixation::SVector{2, Float32},
                    initial_scene::WorldState;
                    overlap_density=0.85,
                    target_rf_count=128,
                    fovea_radius_ratio=0.03,
                    fovea_rf_fraction=0.45)

    # 1. Initialize Receptive Field Geometry on GPU
    fields_py = jax_script[].generate_receptive_fields(
        image_shape,
        overlap_density,
        target_rf_count,
        fovea_radius_ratio,
        fovea_rf_fraction
    )

    # 2. Allocate fixed host buffers in Julia
    fixation_buf = MVector{2, Float32}(initial_fixation)
    scene_buf = scene_to_array(initial_scene)

    # 3. Create zero-copy Python views
    fixation_buf_py = Py(fixation_buf)
    scene_buf_py = Py(scene_buf)

    RFGraphics(fields_py, fixation_buf, scene_buf, fixation_buf_py, scene_buf_py)
end

function sync_scene(graphics::RFGraphics, 
                    scene::WorldState,
                    fixation::S2V)
    # Sync object buffer
    sync_scene(graphics, scene)
    
    # Sync fixation buffer
    graphics.fixation_buf[1] = Float32(fixation[1])
    graphics.fixation_buf[2] = Float32(fixation[2])
    
    return graphics
end

function sync_scene(graphics::RFGraphics, 
                    scene::WorldState)
    n = length(scene.objects)
    n === length(graphics.scene_buf) || error("Scene and buffer length missmatch")

    @inbounds for i = 1:n
        obj = scene.objects[i]
        x, y = obj.pos
        graphics.scene_buf[i] = [x,y,        # position
                                 obj.radius, # radius
                                 0,          # shape - circle
                                 1, 1, 1]    # rgb
    end

    return graphics
end

function field_predict(graphics::RFGraphics)
    n_points = length(graphics.scene_buf)
    outputs_py = jax_script[].sync_and_sample(
        graphics.fixation_buf_py,
        graphics.fields_py,
        graphics.scene_buf_py,
        n_points,
        rand(Int32)
    )
end

struct ReceptiveFields <: Distribution{Py} end

const receptive_fields = ReceptiveFields()

function Gen.random(::ReceptiveFields, graphics::RFGraphics)
    field_predict(graphics)
end

(::ReceptiveFields)(graphics::RFGraphics) = random(receptive_fields, graphics)

function Gen.logpdf(::ReceptiveFields, x::Py, graphics::RFGraphics)
    n = length(graphics.scene_buf)
    py = jax_script[].sync_and_logpdf(x,
                                      graphics.fixation_buf_py,
                                      graphics.fields_py,
                                      graphics.scene_buf_py,
                                      n)
    pyconvert(Float64, py)
end

# TODO: we could implement for gradient methods but maybe later
# function logpdf_grad(::ReceptiveFields, x::Ref{Py}, rfs::Ref{Py}, scene::Ref{Py})
# end

Gen.has_argument_grads(::ReceptiveFields) = (false,)
Gen.has_output_grad(::ReceptiveFields) = false
Gen.is_discrete(::ReceptiveFields) = false
