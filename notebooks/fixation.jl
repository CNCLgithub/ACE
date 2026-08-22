### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ae4cb95e-9c2b-11f1-b71e-69c35553de55
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	
	using Gen
	using Luxor
	using PlutoUI
	using Random
	using Printf
	# using StatProfilerHTML
	
	using Revise
	using GenRFS
	using ACE
end

# ╔═╡ a74b2f5c-363b-4022-b575-68a8c9564625
html"""
<style>
    @media screen {
        main {
            margin: 0 auto;
            max-width: 3000px;
            padding-left: max(100px, 10%);
            padding-right: max(100px, 10%);
        }
    }
	pluto-output {
    font-size: 1.2em; /* Adjust base text size */
    font-family: "Inter";
	}

pluto-output h1 {
    font-size: 2.5rem; /* Adjust header sizes */
	font-family: "Inter";
}

pluto-output h2 {
    font-size: 3.0rem;
}

cm-editor .cm-scroller,
.cm-editor .cm-content {
    font-family: "Fira Code", monospace !important;
    font-size: 18px !important; /* Adjust size here */
}
</style>
""" 


# ╔═╡ ccc3c256-6daa-43db-b3a9-e3a593c80c2a
"""
    sample_trial()

Creates a trial with 3 objects randomly moving about.
"""
function sample_trial()
    istate = WorldState([
        # first 4 are targets
        Disc(S2V( 150,   20), S2V(1, 0), 10.0),
        Disc(S2V( 100,   0), S2V(-1, 0), 10.0),
        Disc(S2V( -80, 130), S2V(0.5, -1), 10.0),
        Disc(S2V(   0,   0), S2V(1, 0), 10.0),
        # Distractors
        Disc(S2V(-100,   0), S2V(-1, 0), 10.0),
        Disc(S2V( -30,  50), S2V(0.5, -1), 10.0),
        Disc(S2V(  45,-100), S2V(1, 0), 10.0),
        Disc(S2V( 100, -80), S2V(-1, 0), 10.0),
    ])

    motion = BrownianVel(;jitter=0.1)
    graphics = RFGraphics((400, 400), S2V32(0, 0), istate;
                          target_rf_count=256,
                          fovea_radius_ratio=0.075,
                          fovea_rf_fraction=0.25)
    wm = WorldModel(motion, graphics)

    time = 120
    # Simulate true physical object trajectory forward in time
    gt_states = Vector{WorldState}(undef, time)
    curr_state = istate
    for t in 1:time
        curr_state = ACE.resolve_motion(wm.motion, curr_state)
        gt_states[t] = curr_state
    end

    return (gt_states, time, istate, wm)
end

# ╔═╡ 23563c80-767c-47a9-9b07-81fd215a23c7
function test_decision()
    (gt_states, time, istate, wm) = sample_trial()

    vis = PFPerception(
        PFProtocol(; particles=10),
        (0, istate, wm),
        choicemap()
    )
    vstate = PFPerceptionState(vis)
    perception = MentalModule(vis, vstate)

    decision_making = MentalModule(TargetDesignation(; ntarget = 4))

    attention = MentalModule(AdaptiveComputation(
        base_steps = 9,
        buffer_size = 500,
        nns = 20,
        itemp = 3.0,
        load = 20,
        load_m = 10.0,
        load_x0 = -50.0,
        vis_partition=WMPartition{ACE.STrace}(),
        cog_partition=WMPartition{ACE.PiTrace}(),
    ))

    fixation = MentalModule(GDFixation(; 
                                       eta_saccade = 0.001,
                                       lr = 1.0,
                                         momentum = 0.9,
                                        num_steps = 100,
                                        sigma_fovea = 5.0,
                                        gamma = 0.9,
                                        lambda_l2 = 0.0001,
                                        lambda_smooth = 0.0005
))

    avg_runtime = zeros(4)
    snapshots = Vector{Drawing}(undef, time)

    for t = 1:time
        # 1. Get current agent fixation coordinate
        _, fstate = mparse(fixation)
        current_fix = S2V(fstate.fixation[1], fstate.fixation[2])

        # 2. Render ground truth receptive field observation conditioned on current fixation
        ACE.sync_scene(wm.graphics, gt_states[t], current_fix)
        obs_sample = ACE.field_predict(wm.graphics)

        # 3. Form combined observation + fixation constraint choicemap for step t
        obs_t = choicemap(
            (:states => t => :observe, obs_sample),
            (:states => t => :fixation, current_fix)
        )

        # 4. Step cognitive modules across time t = 1, 2, ..., time

        stats = @timed ACE.step_module!(perception, t, obs_t)
        avg_runtime[1] += stats.time
        stats = @timed ACE.step_module!(decision_making, t, perception)
        avg_runtime[2] += stats.time
        # @profile_html ACE.step_module!(attention, t, perception, decision_making)
        stats = @timed ACE.step_module!(attention, t, perception, decision_making)
        avg_runtime[3] += stats.time
        stats = @timed ACE.step_module!(fixation, t, perception, attention)
        avg_runtime[4] += stats.time

        # 5. Render visualizations
        inferred = paint_state(perception, false)
        inferred = paint_state(decision_making, inferred, false)
        inferred = paint_state(attention, inferred)

        snapshots[t] = hcat(
            paint_state(gt_states[t], wm, true),
            paint_state(wm.graphics, gt_states[t]; mode=:mean, show_objects=false, back_color="black"),
            inferred;
            hpad=10
        )
    end

    @printf "Average runtime: V %.2fms | D %.2fms \n | A %.2fms | F %.2fms |" ((avg_runtime ./ time .* 1000)...)
    return snapshots
end;

# ╔═╡ 9ae4b323-376f-4699-ba8f-f33710365ca8
snapshots = test_decision();

# ╔═╡ b6303774-b928-46f9-becb-05f3393c966f
@bind time_step Slider(1:length(snapshots), default=1, show_value=x->"  Step $x")

# ╔═╡ 24527534-6813-4ed3-9263-d0b381a07912
snapshots[time_step]

# ╔═╡ Cell order:
# ╠═ae4cb95e-9c2b-11f1-b71e-69c35553de55
# ╟─a74b2f5c-363b-4022-b575-68a8c9564625
# ╠═ccc3c256-6daa-43db-b3a9-e3a593c80c2a
# ╠═23563c80-767c-47a9-9b07-81fd215a23c7
# ╠═9ae4b323-376f-4699-ba8f-f33710365ca8
# ╟─b6303774-b928-46f9-becb-05f3393c966f
# ╟─24527534-6813-4ed3-9263-d0b381a07912
