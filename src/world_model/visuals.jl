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
    x, y = wm.dimensions
    d = init_viz!(x, y)
    n = length(ws.objects)
    for i = 1:n
        x = ws.objects[i]
        paint!(x)
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

function paint!(d::Disc, color::S3V = [0, 0, 0],
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
