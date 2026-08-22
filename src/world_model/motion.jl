export MotionModel, BrownianVel


@kwdef struct BrownianVel <: MotionModel
    dimensions::S2V = S2V(400., 400.)
    jitter::Float64 = 1.0
    # TODO
end

function prior_params(model::BrownianVel, prev::WorldState)
    Fill(model.jitter, length(prev.objects))
end

function resolve_motion(model::BrownianVel,
                        prev::WorldState)
    n = length(prev.objects)
    new_objects = Vector{Disc}(undef, n)
    @inbounds for i = 1:n
        obj = prev.objects[i]
        new_pos, new_vel =
            resolve_oob(model.dimensions, obj.pos, obj.vel, obj.radius)
        new_objects[i] = Disc(new_pos, new_vel, obj.radius)
    end
    WorldState(new_objects)
end

function resolve_motion(model::BrownianVel,
                        prev::WorldState,
                        forces::AbstractVector{S2V})
    n = length(prev.objects)
    length(forces) === n || error("Forces length missmatch")

    new_objects = Vector{Disc}(undef, n)
    @inbounds for i = 1:n
        obj = prev.objects[i]
        vx,vy = obj.vel + forces[i]
        vel = S2V(
            clamp(vx, -15.0, 15.0),
            clamp(vy, -15.0, 15.0),
        )
        new_pos, new_vel =
            resolve_oob(model.dimensions, obj.pos, vel, obj.radius)
        new_objects[i] = Disc(new_pos, new_vel, obj.radius)
    end
    WorldState(new_objects)
end


function resolve_oob(dimensions::S2V,
                     pos::S2V,
                     vel::S2V,
                     radius::Float64)
    new_pos = (x, y) = pos + vel
    if abs(x) > 0.5 * dimensions[1] - radius
        vel = vel .* S2V(-1., 1)
        new_pos = pos + vel
    elseif abs(y) > 0.5 * dimensions[2] - radius
        vel = vel .* S2V(1, -1)
        new_pos = pos + vel
    end
    new_pos, vel
end
