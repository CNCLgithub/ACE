### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ ae4cb95e-9c2b-11f1-b71e-69c35553de55
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	
	using Gen
	using Luxor
	using PlutoUI
	using Random
	using Printf
	using Statistics
	
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
    # TODO: Sample the locations of each object randomly and independently.
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
"""
    run_model(model_name, trial)

Runs the model once on the trial. Returns accuracy and wallclock in ms.

Valid model names are:
- `:ace`: The Adaptive Computation guided Eye fixation model
- `:central`: Central fixation model
"""
function run_model(model_name::Symbol, gt_states)
    
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

    fixation = MentalModule(
        if model_name == :ace
            GDFixation(; 
                                           eta_saccade = 0.001,
                                           lr = 1.0,
                                             momentum = 0.9,
                                            num_steps = 100,
                                            sigma_fovea = 5.0,
                                            gamma = 0.9,
                                            lambda_l2 = 0.0001,
                                            lambda_smooth = 0.0005
                      )
        elseif model_name == :central
            CentralFixation()
        else
            error("Unsupported model name")
        end
    )

    total_runtime = 0.0
    for t = 1:time
        # 1. Get current agent fixation coordinate
        current_fix = get_fixation(fixation)

        # 2. Render ground truth receptive field observation conditioned on current fixation
        ACE.sync_scene(wm.graphics, gt_states[t], current_fix)
        obs_sample = ACE.field_predict(wm.graphics)

        # 3. Form combined observation + fixation constraint choicemap for step t
        obs_t = choicemap(
            (:states => t => :observe, obs_sample),
            (:states => t => :fixation, current_fix)
        )

        # 4. Step cognitive modules across time t = 1, 2, ..., time

        stats = @timed begin
            ACE.step_module!(perception, t, obs_t)
            ACE.step_module!(decision_making, t, perception)
            ACE.step_module!(attention, t, perception, decision_making)
            ACE.step_module!(fixation, t, perception, attention)
        end

        total_runtime += stats.time
    end

    acc = tracking_accuracy(decision_making, perception, gt_states[end])

    (acc, total_runtime)
end;

# ╔═╡ a3797b35-a43b-4524-ad9e-6831790e18c7
function compare_models(n_trials = 5, n_runs = 10)
    # 1. Sample n trials
    # 2. Run each model on each trial `n_runs` times
    # 3. Get the accuracy and runtime across these runs
    # 4. Compute the t-test difference between means of the ACE and central fixation models for both accuracy and runtime;
    # HINT: feel free to use Julia's `Statistics` module. 
end

# ╔═╡ Cell order:
# ╠═ae4cb95e-9c2b-11f1-b71e-69c35553de55
# ╟─a74b2f5c-363b-4022-b575-68a8c9564625
# ╠═ccc3c256-6daa-43db-b3a9-e3a593c80c2a
# ╠═23563c80-767c-47a9-9b07-81fd215a23c7
# ╠═a3797b35-a43b-4524-ad9e-6831790e18c7
