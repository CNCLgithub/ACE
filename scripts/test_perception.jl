using Gen
using ACE

function sample_trial()
    istate = WorldState([
        Disc(S2V(-10, 10), S2V(1, 0), 5.0),
        Disc(S2V(10, 10), S2V(-1, 0), 5.0),
        Disc(S2V(0, 10), S2V(0.5, -1), 5.0),
    ])

    motion = BrownianVel()
    graphics = RFGraphics((400, 400), 10.0, 2.0, 0.1, S2V32(0,0), istate)
    wm = WorldModel(motion, graphics)

    time = 10
    tr, _ = generate(time_kernel_unfold, (time, istate, wm))
    choices = get_choices(tr)

    obs = Vector{ChoiceMap}(undef, time)
    for t = 1:time
        obs[t] = choicemap((t => :observe, choices[t => :observe]))
    end

    return (time, istate, wm, obs)
end

function test_perception()
    (time, istate, wm, obs) = sample_trial()

    vis = PFPerception(
        PFProtocol(; particles=10),
        (0, istate, wm),
        choicemap()
    )
    vstate = PFPerceptionState(vis)
    vmod = MentalModule(vis, vstate)
    
	snapshots = Vector{Drawing}(undef, time)
	for t = 1:steps
        ACE.step_module!(vmod, t, obs[t])
		# Visualizations
		drawing = paint_state(agent.perception, true)
		snapshots[t] = drawing
	end
end

test_perception();
