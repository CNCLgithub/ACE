################################################################################
# Trace methods
################################################################################

gen_fn(::WorldModel) = s_model
const STrace = Gen.get_trace_type(s_model)

function get_last_state(tr::STrace)
    t, wm, istate = get_args(tr)
    t == 0 ? istate : last(get_retval(tr))
end

function pretty_state(tr::STrace)
    pretty_state(get_last_state(tr))
end
