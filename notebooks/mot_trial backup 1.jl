### A Pluto.jl notebook ###
# v1.1.0

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

# ╔═╡ ae4cb95e-9c2b-11f1-b71e-69c35553de00
begin
    using Pkg
    # Point this at the ACE project (the directory holding ACE's Project.toml),
    # not at this script's parent: the project must resolve Gen, GenRFS and ACE.
    Pkg.activate(joinpath(@__DIR__, ".."))
    
    using Gen
    using Luxor
    using PlutoUI
    using Random
    using Printf
    using DelimitedFiles
    using CodecZlib  
    using Revise
    
    using GenRFS
    using ACE
end

# ╔═╡ 8b1f4e60-2a9d-4c73-b5e8-1f7a3c9d0e42
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

# ╔═╡ 2a00b05f-4baf-4af5-91fc-00b99a5ac370
md"""
# Playback of a recorded stimulus trajectory

Fairly minimal variant of `fixation.jl`. Instead of sampling a trial from
`BrownianVel`, the ground-truth object positions are read from a recorded
MOT/MIT trajectory converted by `mot_tracks_to_ace.py`
(`ace_traj_NNN.csv.gz`, long format, ACE world units, y up).

The world model is still used for the graphics model (receptive-field
rendering) and as the motion prior handed to perception; only
`gt_states` is replaced by the recorded positions.
"""

# ╔═╡ 135afeb1-d2cf-457c-81eb-35affdeedb8a
md"""
## Configuration

`tracks_dir` is the `--out-dir` passed to `mot_tracks_to_ace.py`; it must
contain `ace_traj_NNN.csv[.gz]` and, preferably, `ace_tracks_meta.csv`.
`stride` thins the 50 Hz trajectory to the step rate used by the model:
`stride = 4` gives 12.5 steps/s, 125 steps over the 10 s trial.
"""

# ╔═╡ 5e85800e-df74-4260-8699-fa1988e7e0b1
config = (
    max_frame = 150,
    tracks_dir = abspath(@__DIR__, "..", "data", "mot", "ace"),
    traj = 2,
    stride = 4,
    ntarget = 3,
    arena = 400,
    particles = 10,
)

# ╔═╡ cf3a05ac-1306-46ef-a4d8-c570b2a8b83b
md"""
## Load the trajectory

`load_ace_tracks` rebuilds `Vector{WorldState}` per trajectory with the stored
per-frame velocities and disc radius. It is inlined below (previously an
`include` of `load_ace_tracks.jl`), so the notebook is self-contained apart
from the generated tables. `.gz` input needs `CodecZlib` in the active
project (`using DelimitedFiles` is in the helper cell itself); plain
`.csv` needs no extra package.
"""

# ╔═╡ 9e4a2b71-5c6f-4d38-a0b9-7f1c3e5d8b26
begin
    function _ace_read_bytes(path::AbstractString)
        if endswith(path, ".gz")
            # Imported here, not in the setup cell, so that a missing CodecZlib
            # only breaks the .gz path and the cell stays re-runnable.
            return transcode(CodecZlib.GzipDecompressor, read(path))
        end
        return read(path)
    end
    
    function _ace_header(path::AbstractString)
        line = first(eachline(IOBuffer(_ace_read_bytes(path))))
        return strip.(split(line, ','))
    end
    
    function _read_ace_table(path::AbstractString)
        return readdlm(IOBuffer(_ace_read_bytes(path)), ',', Float64; skipstart = 1)
    end
    
    function _ace_radius_from_meta_first(files, meta_path)
        if meta_path === nothing
            candidate = joinpath(dirname(files[1]), "ace_tracks_meta.csv")
            meta_path = isfile(candidate) ? candidate : nothing
        end
        if meta_path !== nothing && isfile(meta_path)
            meta, header = readdlm(meta_path, ',', String; header = true)
            idx = findfirst(==("object_radius_units"), vec(header))
            idx === nothing || return parse(Float64, meta[1, idx])
        end
        return 200.0 * 0.35 / 12.5
    end
end

# ╔═╡ 3c7d9a10-8b2e-4c5d-9f1a-6d0b8e2c4a73
"""
    load_ace_tracks(dir; radius = nothing, meta_path = nothing)

Read the per-trajectory tables written by `mot_tracks_to_ace.py`
(`ace_traj_NNN.csv[.gz]`, columns `traj, frame, time_s, object, x, y, vx, vy,
radius, is_target`, coordinates already in ACE world units) and return
`Dict{Int, Vector{WorldState}}`, one entry per stimulus trajectory.

`WorldState` is built per frame as `Disc(pos, vel, radius)`, ordered by the
`object` column. `vel` is taken from the `vx`/`vy` columns, i.e. the forward
difference `pos[t+1] - pos[t]` that `resolve_motion` applies in one step. The
`radius` column is used directly unless the `radius` keyword is given.
"""
function load_ace_tracks(dir::AbstractString; radius = nothing, meta_path = nothing)
    isdir(dir) || error("$dir is not a directory; pass the --out-dir used with mot_tracks_to_ace.py")
    files = sort(filter(f -> occursin(r"^ace_traj_\d+\.csv(\.gz)?$", basename(f)),
                        readdir(dir; join = true)))
    isempty(files) && error("no ace_traj_NNN.csv[.gz] files in $dir")

    if radius === nothing
        radius = _ace_radius_from_meta_first(files, meta_path)
    end

    tracks = Dict{Int, Vector{WorldState}}()
    for file in files
        table = _read_ace_table(file)
        col = Dict(name => i for (i, name) in enumerate(_ace_header(file)))
        for c in ("traj", "frame", "object", "x", "y", "vx", "vy")
            haskey(col, c) || error("$file: missing column $c")
        end

        traj_ids = unique(Int.(table[:, col["traj"]]))
        length(traj_ids) == 1 || error("$file: expected one trajectory, found $(length(traj_ids))")
        traj = traj_ids[1]

        frames = unique(Int.(table[:, col["frame"]]))
        states = Vector{WorldState}(undef, length(frames))
        for (fi, frame) in enumerate(frames)
            rows = table[Int.(table[:, col["frame"]]) .== frame, :]
            order = sortperm(rows[:, col["object"]])
            objects = Vector{Disc}(undef, length(order))
            for (oi, r) in enumerate(order)
                pos = S2V(rows[r, col["x"]], rows[r, col["y"]])
                vel = S2V(rows[r, col["vx"]], rows[r, col["vy"]])
                rad = radius === nothing ? rows[r, col["radius"]] : radius
                objects[oi] = Disc(pos, vel, rad)
            end
            states[fi] = WorldState(objects)
        end
        tracks[traj] = states
    end
    return tracks
end

# ╔═╡ 5d0f462b-2555-434e-959a-b93b76d08ad4
begin
    tracks = load_ace_tracks(config.tracks_dir)
    gt_full = tracks[config.traj]
    frames = 1:config.stride:min(config.max_frame, length(gt_full))
    gt_states = [
        WorldState([
            Disc(obj.pos, Float64(config.stride) .* obj.vel, obj.radius)
            for obj in gt_full[f].objects
        ])
        for f in frames
    ]
    time = length(gt_states)
    step_hz = 50.0 / config.stride

end

# ╔═╡ a39ed78b-8d8d-45d7-ac01-c8fb7a468926
md"""
## World model and graphics

`BrownianVel` is kept as the motion prior (its `dimensions` set the arena used
by out-of-bounds reflection in the generative model, not by the recorded
trajectory). The graphics model is unchanged from `fixation.jl`.
"""

# ╔═╡ a3f563da-6435-4148-b76f-add056cd4074
begin
    motion = BrownianVel(; dimensions = S2V(Float64(config.arena), Float64(config.arena)),
                         jitter = 2.5)
    
    graphics = RFGraphics((config.arena, config.arena), S2V32(0, 0), gt_states[1];)
    
    wm = WorldModel(motion, graphics)
    
    max_reach = maximum(maximum(abs.([o.pos[1], o.pos[2]]) for o in s.objects)
                        for s in gt_states)
end

# ╔═╡ 4e023fc2-bd04-47db-aafc-af79187bfdd5
md"""
Maximum |coordinate| over the loaded steps: $(round.(max_reach, digits = 1)) units
(arena half-extent $(config.arena ÷ 2)).
"""

# ╔═╡ 011c008c-6ef3-4efe-8227-f1578b35f1c8
md"""
## Agent modules

Identical to `fixation.jl`, with `ntarget` taken from the configuration
(the converter marks objects 1..3 as targets for this dataset).
"""

# ╔═╡ 6e57324b-e0fc-425f-9edf-9641c7c20abd
begin
    vis = PFPerception(
        PFProtocol(; particles = config.particles),
        (0, gt_states[1], wm),
        choicemap()
    )
    vstate = PFPerceptionState(vis)
    perception = MentalModule(vis, vstate)
    
    decision_making = MentalModule(TargetDesignation(; ntarget = config.ntarget))
    
    attention = MentalModule(AdaptiveComputation(
        base_steps = 9,
        buffer_size = 500,
        nns = 20,
        itemp = 3.0,
        load = 20,
        load_m = 10.0,
        load_x0 = -50.0,
        vis_partition = WMPartition{ACE.STrace}(),
        cog_partition = WMPartition{ACE.PiTrace}(),
    ))
    
    fixation = MentalModule(GDFixation(;
        eta_saccade = 0.001,
        lr = 5.0,
        momentum = 0.9,
        num_steps = 100,
        sigma_fovea = 50.0,
        gamma = 0.9,
        lambda_l2 = 0.0001,
        lambda_smooth = 0.0005,
    ))
end

# ╔═╡ c9f2a0b1-0f5f-4b7f-9d0e-6a1c9e0b7d31
md"""
## Run the model

The loop is unchanged: fixate, render the receptive-field observation at the
current fixation, and step perception → target designation → attention →
fixation. `gt_states[t]` here is a recorded frame rather than a sampled one.

The snapshot carries four panes: ground truth, the receptive-field observation,
the inferred world with attention markers, and the task-relevance field that the
fixation controller optimises. The same quantities are also logged per tick in
`fix_log` (see below) so that a failure to saccade can be read off a trace as
well as a frame.
"""

# ╔═╡ d81c4a55-7c83-4e2b-9a55-2f5f4c2b6e10
function run_trial(gt_states, wm, perception, decision_making, attention, fixation)
    time = length(gt_states)
    avg_runtime = zeros(4)
    snapshots = Vector{Drawing}(undef, time)
    # Columns: R(fix), R(prop), gain, |prop - fix|, avg_load
    fix_log = zeros(time, 5)

    for t in 1:time
        _, fstate = mparse(fixation)
        current_fix = S2V(fstate.fixation[1], fstate.fixation[2])

        ACE.sync_scene(wm.graphics, gt_states[t], current_fix)
        obs_sample = ACE.field_predict(wm.graphics)

        obs_t = choicemap(
            (:states => t => :observe, obs_sample),
            (:states => t => :fixation, current_fix)
        )

        stats = @timed ACE.step_module!(perception, t, obs_t)
        avg_runtime[1] += stats.time
        stats = @timed ACE.step_module!(decision_making, t, perception)
        avg_runtime[2] += stats.time
        stats = @timed ACE.step_module!(attention, t, perception, decision_making)
        avg_runtime[3] += stats.time
        stats = @timed ACE.step_module!(fixation, t, perception, attention)
        avg_runtime[4] += stats.time

        fprot, fstate = mparse(fixation)
        _, astate = mparse(attention)
        if !isempty(fstate.targets)
            sigma = fprot.sigma_fovea
            fix_x = Float64(fstate.fixation[1])
            fix_y = Float64(fstate.fixation[2])
            pro_x = Float64(fstate.proposal[1])
            pro_y = Float64(fstate.proposal[2])
            fix_log[t, 1] = ACE.relevance_density(fstate.targets, fstate.weights, sigma, fix_x, fix_y)
            fix_log[t, 2] = ACE.relevance_density(fstate.targets, fstate.weights, sigma, pro_x, pro_y)
            fix_log[t, 3] = fstate.gain
            fix_log[t, 4] = hypot(pro_x - fix_x, pro_y - fix_y)
            fix_log[t, 5] = astate.avg_load
        end

        inferred = paint_state(perception, false)
        inferred = paint_state(decision_making, inferred, false)
        inferred = paint_state(attention, inferred)
        relevance = paint_state(fixation, true)

        snapshots[t] = hcat(
            paint_state(gt_states[t], wm, true),
            paint_state(wm.graphics, gt_states[t]; mode = :mean,
                        show_objects = false, back_color = "black"),
            inferred,
            relevance;
            hpad = 10
        )
    end

    per_step = avg_runtime ./ time .* 1000
    @printf "Average runtime: V %.2fms | D %.2fms | A %.2fms | F %.2fms\n" per_step...
    return (snapshots = snapshots, fix_log = fix_log)
end

# ╔═╡ e0b3d1c9-5f2a-4a4b-8c1d-3e7a9b6c5f42
result = run_trial(gt_states, wm, perception, decision_making, attention, fixation);
snapshots = result.snapshots;
fix_log = result.fix_log;

# ╔═╡ b4d8e6a2-7c31-4f9e-8a2d-1c5e0f3b9a62
md"""
### Fixation trace

`R(fix)` and `R(prop)` are the task-relevance density at the current fixation and
at the fixation the optimiser proposes; their difference is the gain that
`eta_saccade` gates. A flat `R(fix)` with a rising `R(prop)` means the optimiser
wants to move but the gate is refusing. Two consecutive small `R(prop)` values
mean the field itself has no competing mode, which is a relevance or perception
problem rather than a controller problem.
"""

# ╔═╡ a2f6c9d4-3e8b-4a17-b5c0-6d9f2e7b1c84
begin
    println("   t    t[s]   R(fix)   R(prop)     gain    |df|    load")
    for t in 1:size(fix_log, 1)
        @printf "%4d  %6.2f  %8.5f  %8.5f  %8.5f  %6.1f  %5.2f\n" t (t - 1) / step_hz fix_log[t, 1:5]...
    end
end

# ╔═╡ f2c9b7a3-1d4e-4f5a-8b6c-9d0e1f2a3b4c
@bind time_step Slider(1:length(snapshots), default = 1,
                       show_value = x -> @sprintf("  step %d / %d  (t = %.2f s)",
                                                  x, length(snapshots), (x - 1) / step_hz))

# ╔═╡ a7f3c2d1-4b5e-4f6a-9c8d-2e1f0a3b5c7d
snapshots[time_step]

# ╔═╡ b3e8f1a2-6c7d-4e5f-8a9b-1c2d3e4f5a6b
"""
    save_snapshots_as_mp4(snapshots, output_file="track_playback.mp4";
                          fps=24, temp_dir=mktempdir())

Exports each frame via Julia's `Base.show` MIME interface and compiles to MP4
with ffmpeg.
"""
function save_snapshots_as_mp4(snapshots, output_file = "track_playback.mp4";
                               fps = 24, temp_dir = mktempdir())
    @info "Exporting $(length(snapshots)) frames to $temp_dir..."

    sample = first(snapshots)
    use_png = showable(MIME("image/png"), sample)
    mime_type = use_png ? MIME("image/png") : MIME("image/svg+xml")
    ext = use_png ? "png" : "svg"

    for (i, d) in enumerate(snapshots)
        frame_path = joinpath(temp_dir, @sprintf("frame_%04d.%s", i, ext))
        open(frame_path, "w") do io
            show(io, mime_type, d)
        end
    end

    input_pattern = joinpath(temp_dir, "frame_%04d.$ext")
    @info "Compiling video with ffmpeg -> $output_file"
    run(`ffmpeg -y -framerate $fps -i $input_pattern -c:v libx264 -pix_fmt yuv420p -crf 18 $output_file`)

    @info "Animation saved to $output_file"
    return output_file
end

# ╔═╡ 4c8e2a17-6d3b-4f90-a2c5-8b7e1f0d9c36
save_snapshots_as_mp4(snapshots, "track_playback_$(config.traj).mp4")

# ╔═╡ Cell order:
# ╟─ae4cb95e-9c2b-11f1-b71e-69c35553de00
# ╟─8b1f4e60-2a9d-4c73-b5e8-1f7a3c9d0e42
# ╟─2a00b05f-4baf-4af5-91fc-00b99a5ac370
# ╟─135afeb1-d2cf-457c-81eb-35affdeedb8a
# ╠═5e85800e-df74-4260-8699-fa1988e7e0b1
# ╟─cf3a05ac-1306-46ef-a4d8-c570b2a8b83b
# ╟─3c7d9a10-8b2e-4c5d-9f1a-6d0b8e2c4a73
# ╟─9e4a2b71-5c6f-4d38-a0b9-7f1c3e5d8b26
# ╟─5d0f462b-2555-434e-959a-b93b76d08ad4
# ╟─a39ed78b-8d8d-45d7-ac01-c8fb7a468926
# ╠═a3f563da-6435-4148-b76f-add056cd4074
# ╟─4e023fc2-bd04-47db-aafc-af79187bfdd5
# ╟─011c008c-6ef3-4efe-8227-f1578b35f1c8
# ╠═6e57324b-e0fc-425f-9edf-9641c7c20abd
# ╟─c9f2a0b1-0f5f-4b7f-9d0e-6a1c9e0b7d31
# ╠═d81c4a55-7c83-4e2b-9a55-2f5f4c2b6e10
# ╠═e0b3d1c9-5f2a-4a4b-8c1d-3e7a9b6c5f42
# ╟─f2c9b7a3-1d4e-4f5a-8b6c-9d0e1f2a3b4c
# ╠═a7f3c2d1-4b5e-4f6a-9c8d-2e1f0a3b5c7d
# ╟─b3e8f1a2-6c7d-4e5f-8a9b-1c2d3e4f5a6b
# ╠═4c8e2a17-6d3b-4f90-a2c5-8b7e1f0d9c36
