export init_viz!,
    paint!,
    paint_state,
    paint_masks

import Colors

function init_viz!(w::Float64, h::Float64,
                   back_color="black")
    d = Drawing(w, h, :svg)
    origin()
    background(back_color)
    return d
end

function paint_state(ws::WorldState, wm::WorldModel, ret_finish=true)
    x, y = wm.motion.dimensions
    d = init_viz!(x, y)
    n = length(ws.objects)
    for i = 1:n
        x = ws.objects[i]
        paint!(x, S3V(1,1,1))
    end
	ret_finish && finish()
	return d
end

# function paint_masks(masks::AbstractArray{Mask}, wm::WorldModel)
#     x, y = wm.dimensions
#     d = init_viz!(x, y)
#     n = length(masks)
#     for i = 1:n
#         paint!(masks[i])
#     end
# 	finish()
# 	return d
# end


# function paint!(shp::Rectangle, pos::S2V, hue::Float64,
#                 opacity=1.0, size_scale::Float64=1.0)
#     # TODO: fix color space
#     color = Colors.MSC(hue)
#     point = Luxor.Point(pos[1], -pos[2])
#     w = 2*shp.hw*size_scale
#     h = 2*shp.hh*size_scale
#     # Isolate the rotation for a specific object
#     @layer begin
#         setopacity(opacity)
#         sethue(color)
#         translate(point)
#         rotate(shp.angle) 
#         Luxor.box(Luxor.Point(0,0), w, h, :fill)
#     end
#     return nothing
# end

function paint!(d::Disc, color::AbstractVector = [0, 0, 0],
                opacity=1.0, size_scale::Float64=1.0)
    color = Colors.RGB(color...)
    point = Luxor.Point(d.pos[1], -d.pos[2])
    radius = size_scale * d.radius
    @layer begin
        setopacity(opacity)
        sethue(color)
        translate(point)
        Luxor.circle(Luxor.Point(0, 0), radius, :fill)
    end
    return nothing
end

# function paint!(mask::Mask, opacity=1.0)
#     color = Colors.MSC(rad2deg(mask.hsv))
#     point = Luxor.Point(mask.pos[1], -mask.pos[2])
#     w, h = mask.extents
#     @layer begin
#         setopacity(opacity)
#         sethue(color)
#         Luxor.box(point,
#                   mask.fill_ratio*w,
#                   mask.fill_ratio*h,
#                   :fill)
#         Luxor.box(point, w, h, :stroke)
#     end
#     return nothing
# end


"""
    paint!(rf::AbstractVector{<:Real}, color::S3V; opacity=1.0, mode=:fill, line_width=1.0)
"""
function paint!(rf::AbstractVector{<:Real},
                color::S3V = [0.0, 0.0, 0.0];
                opacity::Float64 = 1.0,
                mode::Symbol = :fill,
                line_width::Float64 = 1.0)
    point = Luxor.Point(rf[1], -rf[2])
    radius = Float64(rf[3])
    c = Colors.RGB(clamp.(color, 0.0, 1.0)...)
    
    @layer begin
        setopacity(opacity)
        sethue(c)
        if mode === :fill
            Luxor.circle(point, radius, :fill)
        elseif mode === :stroke
            setline(line_width)
            Luxor.circle(point, radius, :stroke)
        elseif mode === :fillstroke
            setline(line_width)
            Luxor.circle(point, radius, :fillstroke)
        end
    end
    return nothing
end

"""
    paint_receptive_fields!(graphics::RFGraphics; mode=:mean, fill_opacity=0.65, stroke_opacity=0.35)
"""
function paint_receptive_fields!(graphics::RFGraphics;
                                mode::Symbol = :mean,
                                fill_opacity::Float64 = 0.7,
                                stroke_opacity::Float64 = 0.3)
    np = pyimport("numpy")
    jax = pyimport("jax")

    # Geometry on CPU
    rfs_np = np.array(graphics.fields_py)
    rfs = pyconvert(Matrix{Float32}, rfs_np)
    fix_x, fix_y = graphics.fixation_buf
    n_rf = size(rfs, 1)

    # Evaluate forward model
    n_points = length(graphics.scene_buf)
    fixation_np = np.frombuffer(graphics.fixation_buf_py, dtype=np.float32)
    objects_np = np.frombuffer(graphics.scene_buf_py, dtype=np.float32).reshape((n_points, 7))

    means_jax, vars_jax = jax_script[].predict_rf_stats(
        jax.device_put(fixation_np),
        graphics.fields_py,
        jax.device_put(objects_np)
    )

    means = pyconvert(Matrix{Float32}, np.array(means_jax))
    vars  = pyconvert(Matrix{Float32}, np.array(vars_jax))

    for i in 1:n_rf
        rf_pos_r = [rfs[i, 1] + fix_x, rfs[i, 2] + fix_y, rfs[i, 3]]

        if mode === :mean
            rgb = S3V(means[i, 1], means[i, 2], means[i, 3])
            # Filled receptive field circle
            paint!(rf_pos_r, rgb; opacity=fill_opacity, mode=:fill)
            # Subtle boundary edge
            if stroke_opacity > 0.0
                paint!(rf_pos_r, S3V(1,1,1); opacity=stroke_opacity, mode=:stroke, line_width=1.0)
            end

        elseif mode === :variance
            # Variance visualization (RGB variance directly or scaled)
            var_rgb = S3V(vars[i, 1], vars[i, 2], vars[i, 3])
            paint!(rf_pos_r, var_rgb; opacity=fill_opacity, mode=:fill)
            if stroke_opacity > 0.0
                paint!(rf_pos_r, var_rgb; opacity=stroke_opacity, mode=:stroke, line_width=1.0)
            end
        end
    end
    return nothing
end

"""
    paint_state(graphics::RFGraphics, scene::WorldState; mode=:mean, show_objects=false, canvas_dims=(400.0, 400.0), back_color="black", ret_finish=true)
"""
function paint_state(graphics::RFGraphics,
                     scene::WorldState;
                     mode::Symbol = :mean,
                     show_objects::Bool = false,
                     canvas_dims::Tuple{Float64, Float64} = (400.0, 400.0),
                     back_color::String = "black",
                     fill_opacity::Float64 = 0.7,
                     ret_finish::Bool = true)
    sync_scene(graphics, scene)
    
    d = init_viz!(canvas_dims[1], canvas_dims[2], back_color)
    
    if show_objects
        for obj in scene.objects
            paint!(obj, S3V(0.8, 0.8, 0.8), 0.3)
        end
    end

    paint_receptive_fields!(graphics; mode=mode, fill_opacity=fill_opacity)

    ret_finish && finish()
    return d
end
