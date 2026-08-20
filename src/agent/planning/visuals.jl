
function paint_state(m::MentalModule{TargetDesignation},
                     drawing, ret_finish=true)
    paint_target_selections!(drawing, m)
    ret_finish && finish()
    return drawing
end

function paint_target_selections!(drawing, m)
    # Draw a rectangle around the targets
    # reflect confidence (the retval of the PiTraces) in terms of opacity (more bold => more confident)
end
