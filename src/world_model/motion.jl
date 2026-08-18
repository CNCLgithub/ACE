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
    for i = 1:n
        obj = prev.objects[i]
        new_pos, new_vel =
            resolve_oob(model.dimensions, obj.pos, vel, obj.radius)
        new_objects = Disc(new_pos, new_vel, obj.radius)
    end
    WorldState(new_objects)
end

function resolve_motion(model::BrownianVel,
                        prev::WorldState,
                        forces::AbstractVector{S2V})
    n = length(prev.objects)
    length(forces) === n || error("Forces length missmatch")

    new_objects = Vector{Disc}(undef, n)
    for i = 1:n
        obj = prev.objects[i]
        # Apply random jitter
        if !isnothing(jitter)
            vx,vy = obj.vel + forces[i]
            vel = S2V(
                clamp(vx, -15.0, 15.0),
                clamp(vy, -15.0, 15.0),
            )
        end
        new_pos, new_vel =
            resolve_oob(model.dimensions, obj.pos, vel, obj.radius)
        new_objects = Disc(new_pos, new_vel, obj.radius)
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

function resolve_motion(model::BilliardBrownian,
                        state::WorldState,
                        jitter::Union{<:AbstractArray{S2V}, Nothing} = nothing)

    nstatic = length(state.static)
    ndynamic = length(state.dynamic)

    !isnothing(jitter) &&
        length(jitter) !== ndynamic &&
            error("jitter to state length missmatch")

    new_dynamic = DynamicState(state.dynamic)


    for i = 1:ndynamic
        obj, pos, vel = state.dynamic[i]
        # Apply random jitter
        if !isnothing(jitter)
            vx,vy = vel + jitter[i]
            vel = S2V(
                clamp(vx, -15.0, 15.0),
                clamp(vy, -15.0, 15.0),
            )
        end
        new_pos, new_vel =
            scan_for_collision(model, obj, pos, vel, state.static)

        new_pos, new_vel = resolve_oob(model, obj, pos, new_vel)
        update_position!(new_dynamic, i, new_pos)
        update_velocity!(new_dynamic, i, new_vel)
    end

    WorldState(new_dynamic, state.static)
end
