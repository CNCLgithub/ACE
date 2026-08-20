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
function sample_trial()
    istate = WorldState([
        Disc(S2V(-10, 10), S2V(1, 0), 10.0),
        Disc(S2V(10, 10), S2V(-1, 0), 10.0),
        Disc(S2V(0, 10), S2V(0.5, -1), 10.0),
    ])

    motion = BrownianVel()
    graphics = RFGraphics((400, 400), S2V32(0, 0), istate,
                          overlap_density=1.0,
                          target_rf_count=128,
                          fovea_radius_ratio=0.10,
                          fovea_rf_fraction=0.35)
                         
    wm = WorldModel(motion, graphics)

    time = 30
    tr, _ = generate(s_model, (time, istate, wm))
    choices = get_choices(tr)

    obs = Vector{ChoiceMap}(undef, time)
    for t = 1:time
        obs[t] = choicemap((:states => t => :observe, choices[:states => t => :observe]))
    end

    states = get_retval(tr)
    return (states, time, istate, wm, obs)
end;

# ╔═╡ fff7b806-f785-45f4-982f-03d64aaa502f
function test_decision()
    (gt_states, time, istate, wm, obs) = sample_trial()

    vis = PFPerception(
        PFProtocol(; particles=10),
        (0, istate, wm),
        choicemap()
    )
    vstate = PFPerceptionState(vis)
    perception = MentalModule(vis, vstate)

    decision_making = MentalModule(TargetDesignation(;ntarget = 1))
    
	snapshots = Vector{Drawing}(undef, time)
	for t = 1:time
        ACE.step_module!(perception, t, obs[t])
        ACE.step_module!(decision_making, t, perception)
        @show decision_expectation(decision_making)
        # Visualizations
        inferred = paint_state(perception, false)
        inferred = paint_state(decision_making, inferred)
        
        snapshots[t] = hcat(
            paint_state(gt_states[t], wm, true), # gt state
            # Receptive Fields colored by Mean RGB
            paint_state(wm.graphics, gt_states[t]; mode=:mean, show_objects=false, back_color="black"),
            # Receptive Fields colored by Variance
            paint_state(wm.graphics, gt_states[t]; mode=:variance, show_objects=false, back_color="black"),
            # Inferred states
            inferred;
            hpad=10
        )
        
        end
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
# ╠═fff7b806-f785-45f4-982f-03d64aaa502f
# ╠═9ae4b323-376f-4699-ba8f-f33710365ca8
# ╟─b6303774-b928-46f9-becb-05f3393c966f
# ╟─24527534-6813-4ed3-9263-d0b381a07912
