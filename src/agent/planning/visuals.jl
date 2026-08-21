
function paint_state(m::MentalModule{TargetDesignation},
                     drawing, ret_finish=true)
    paint_target_selections!(drawing, m)
    ret_finish && finish()
    return drawing
end

function paint_target_selections!(drawing,
                                 m::MentalModule{TargetDesignation};
                                 box_size::Float64 = 34.0,
                                 target_color = Colors.RGB(0.25, 1.00, 0.25),
                                 temp::Float64 = 10.0)
    protocol, state = mparse(m)
    isempty(state.chain) && return nothing

    ntraces = length(state.chain)
    ntargets = protocol.ntarget

    # Aggregate target positions and marginal target probabilities across traces
    for tr in state.chain
        ws = get_last_state(tr)

        # Compute confidence / expected reward for this trace
        score_val = expected_reward_td_conf(protocol, tr)
        conf = clamp(exp(score_val), 0.1, 1.0)

        for i = 1:protocol.ntarget
            obj = ws.objects[i]
            pos = obj.pos
            point = Point(pos[1], -pos[2])

            # Modulate opacity & boldness by per-object target probability & overall trace confidence
            effective_alpha = clamp(conf * (1.5 / ntraces), 0.0, 1.0)

            if effective_alpha > 0.05
                @layer begin
                    sethue(target_color)
                    setopacity(effective_alpha)
                    setline(2.0 + 2.0 * conf)
                    
                    # Draw target bounding box
                    box(point, box_size, box_size, :stroke)
                    
                    # Subtle corner/inner fill for high confidence
                    if conf > 0.6
                        setopacity(effective_alpha * 0.15)
                        box(point, box_size, box_size, :fill)
                    end
                end
            end
        end
    end
    return nothing
end
