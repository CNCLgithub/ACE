export time_kernel

@gen (static) function force_prior(var::Float64)
    dx ~ normal(0., var)
    dy ~ normal(0., var)
    result::S2V = S2V(dx, dy)
    return result
end

@gen (static) function time_kernel(t::Int,
                                   prev::WorldState,
                                   wm::WorldModel)
    # predict motion
    forces ~ Gen.Map(force_prior)(prior_params(wm.motion, prev))
    next::WorldState = resolve_motion(wm.motion, prev, forces)
    # predict receptive field statistics
    graphics = sync_scene(wm.graphics, next)
    # observe receptive fields
    observe ~ receptive_fields(graphics)
    return next
end
