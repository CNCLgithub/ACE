export PiTrace

@gen (static) function target_designation(ws::WorldState,
                                          ntargets::Int64)
    # Each target is an isomorphic element
    # Distractors summarized as PPP
    elems = select_targets(ws, ntargets)
    # sample a random set based on the current object states
    xs ~ rfs_s2v(elems)
    return xs
end

const PiIR = Gen.get_ir(target_designation)
const PiTrace = Gen.get_trace_type(target_designation)

function extract_rfs_subtrace(trace::PiTrace)
    # StaticIR names and nodes
    rfs_node = PiIR.call_nodes[1] # :xs
    rfs_field = Gen.get_subtrace_fieldname(rfs_node)
    getproperty(trace, rfs_field)
end
