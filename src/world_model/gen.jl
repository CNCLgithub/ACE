export s_model

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
    # Naive prior over fixations
    fixation ~ normal_s2v(S2V(0.0, 0.0), 300.0)
    # predict receptive field statistics
    graphics = sync_scene(wm.graphics, next, fixation)
    # observe receptive fields
    observe ~ receptive_fields(graphics)
    return next
end


@gen (static) function s_model(t::Int,
                               istate::WorldState,
                               wm::WorldModel)
    states ~ Unfold(time_kernel)(t, istate, wm)
    return states
end
