
################################################################################
# Ancestral Proposals
################################################################################

function object_ancestral_proposal(trace::WMTrace,
                                   idx::Int,
                                   back::Int = 3)
    t, _... = get_args(trace)
    t_range = max(1, t-back):t
    selection = select(
        (:states => ti => :jitter => idx for ti = t_range)...,
    )
    new_trace, w, _ = regenerate(trace, selection)

    if isinf(w) || isnan(w)
        find_inf_scores(new_trace)
        error("Invalid score from ancestral proposal")
    end
    (new_trace, w)
end
