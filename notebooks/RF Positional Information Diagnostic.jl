### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 20000000-0000-0000-0000-000000000003
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))

    using ACE
    using Luxor
    using Random
    using Printf
    using Statistics
    using Colors: RGB

    Random.seed!(1)
end

# ╔═╡ 20000000-0000-0000-0000-000000000001
md"""
# Positional information in the receptive-field likelihood

**Question.** The inferred cloud drifts while the observed mosaic still contains the objects. Does the observation density constrain object *position*, or only *presence*?

**Why it may not.** A disc of radius $r$ lying strictly inside a field of radius $\rho$ covers $\pi r^2$ whatever its position inside that field. So

$$\mu = \frac{A}{\pi\rho^2} = \left(\frac{r}{\rho}\right)^2$$

is independent of where the disc sits. All fields in one ring share $\rho$, so every field containing the disc reports the same number, and the density is flat in position except on the annulus of width $2r$ where the disc crosses a field boundary.

**Two tests.** A line sweep of the log-likelihood through an object's true position, and a frame-wide map of the same quantity. Both are computed for the linear Gaussian readout and for the log readout.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000002
md"""
## Setup

Geometry comes from ACE; coverage is computed analytically by circle intersection, so nothing here depends on `predict_rf_stats!`.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000004
begin
    const ARENA = 400.0
    const OBJ_RADIUS = 15.0
    const SIGMA = 0.1
    const SIGMA_Y = 0.4
    const BETA_LOG = 1e-3
    const OUT_DIR = @__DIR__
end

# ╔═╡ 20000000-0000-0000-0000-000000000005
md"""
## Geometry

Field centres are relative to the fixation point (frame centre). If the ACE keyword name differs in your tree, change this cell only.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000006
begin
    geom = ACE.generate_receptive_fields(
        (Int(ARENA), Int(ARENA));
        fovea_radius_ratio = 0.07)

    centers = [(Float64(c.x), Float64(c.y)) for c in geom.fields]
    radii = Float64.(map(c -> c.r, geom.fields))
    nfields = length(centers)

    println("fields: ", nfields)
    println("radius range: ", round(minimum(radii), digits = 1), " to ",
            round(maximum(radii), digits = 1))
end

# ╔═╡ 20000000-0000-0000-0000-000000000007
md"""
## Coverage and the two densities

`circle_overlap` is the exact area of intersection of two circles. `means_for` sums object contributions, divides by field area and clips at 1, matching the forward model. `ll_linear` scores a Gaussian residual on coverage with standard deviation $\sigma = 0.1$; `ll_log` scores a Gaussian residual on $\log(x + \beta)$ with standard deviation $\sigma_y = 0.4$, which is a relative criterion and therefore scale-free across fields.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000008
begin
    function circle_overlap(R::Float64, r::Float64, d::Float64)
        if d >= R + r
            return 0.0
        end
        if d <= abs(R - r)
            return π * min(R, r)^2
        end

        a = r^2 * acos((d^2 + r^2 - R^2) / (2 * d * r))
        b = R^2 * acos((d^2 + R^2 - r^2) / (2 * d * R))
        s = (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
        return a + b - 0.5 * sqrt(s)
    end

    function means_for(objects, centers, radii)
        area = zeros(length(centers))
        for o in objects
            for k in eachindex(centers)
                d = hypot(o.pos[1] - centers[k][1], o.pos[2] - centers[k][2])
                area[k] += circle_overlap(radii[k], o.radius, d)
            end
        end
        mu = area ./ (π .* radii .^ 2)
        return min.(mu, 1.0)
    end

    function ll_linear(obs, pred)
        total = 0.0
        for k in eachindex(obs)
            total -= 0.5 * ((obs[k] - pred[k]) / SIGMA)^2
        end
        return total
    end

    function ll_log(obs, pred)
        total = 0.0
        for k in eachindex(obs)
            r = log(obs[k] + BETA_LOG) - log(pred[k] + BETA_LOG)
            total -= 0.5 * (r / SIGMA_Y)^2
        end
        return total
    end
end

# ╔═╡ 20000000-0000-0000-0000-000000000009
md"""
## Test 1: line sweep

Objects sit at the recorded positions of one frame. The observation is held fixed at the true coverage; one object is translated along a direction and the total log-likelihood is recomputed. The offset at which each curve falls by 1 nat is the length over which the data localise that object.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000010
begin
    truth = [
        (pos = (-60.0, -90.0), radius = OBJ_RADIUS),
        (pos = (110.0, -40.0), radius = OBJ_RADIUS),
        (pos = (20.0, 120.0), radius = OBJ_RADIUS),
        (pos = (150.0, 95.0), radius = OBJ_RADIUS),
    ]

    obs_means = means_for(truth, centers, radii)
    sweep_j = 1
    sweep_dir_raw = (1.0, 0.3)
    sweep_dir = sweep_dir_raw ./ hypot(sweep_dir_raw...)
    offsets = collect(-200.0:4.0:200.0)

    function sweep_curves(truth, j, dir, offsets)
        lin = Float64[]
        lg = Float64[]
        for s in offsets
            moved = copy(truth)
            moved[j] = (
                pos = (truth[j].pos[1] + s * dir[1], truth[j].pos[2] + s * dir[2]),
                radius = truth[j].radius,
            )
            pred = means_for(moved, centers, radii)
            push!(lin, ll_linear(obs_means, pred))
            push!(lg, ll_log(obs_means, pred))
        end
        return lin, lg
    end

    curve_lin, curve_log = sweep_curves(truth, sweep_j, sweep_dir, offsets)

    function half_width(curve, offsets, drop = 1.0)
        peak = maximum(curve)
        idx = findall(c -> c >= peak - drop, curve)
        return offsets[last(idx)] - offsets[first(idx)]
    end

    @printf("1-nat width, linear: %.0f units\n", half_width(curve_lin, offsets))
    @printf("1-nat width, log:    %.0f units\n", half_width(curve_log, offsets))
    @printf("object diameter:     %.0f units\n", 2 * OBJ_RADIUS)
end

# ╔═╡ 20000000-0000-0000-0000-000000000011
begin
    using Base64: base64encode
    using Markdown: Markdown

    function png_view(path)
        data = base64encode(String(read(path)))
        return Markdown.parse("![](data:image/png;base64,$data)")
    end

    function plot_sweep(offsets, curves, path)
        Drawing(900, 520, path)
        background("white")

        w, h = 900.0, 520.0
        ml, mr, mt, mb = 90.0, 40.0, 60.0, 70.0
        x0, x1 = offsets[1], offsets[end]
        vals = reduce(vcat, [c for (c, _) in curves])
        y0 = minimum(vals)
        y1 = maximum(vals)
        if y1 - y0 < 1e-6
            y1 = y0 + 1.0
        end

        X(x) = ml + (x - x0) / (x1 - x0) * (w - ml - mr)
        Y(y) = h - mb - (y - y0) / (y1 - y0) * (h - mt - mb)

        setcolor("gray70")
        setline(1)
        for gy in range(y0, y1, length = 5)
            line(Point(X(x0), Y(gy)), Point(X(x1), Y(gy)), :stroke)
        end

        setcolor("black")
        setline(1.5)
        line(Point(X(x0), Y(y0)), Point(X(x1), Y(y0)), :stroke)
        line(Point(X(x0), Y(y0)), Point(X(x0), Y(y1)), :stroke)

        setcolor("gray40")
        setline(1)
        line(Point(X(0), Y(y0)), Point(X(0), Y(y1)), :stroke)

        setline(2)
        for (k, (c, _)) in enumerate(curves)
            setcolor(k == 1 ? "black" : "gray45")
            p = [Point(X(offsets[i]), Y(c[i])) for i in eachindex(offsets)]
            poly(p, :stroke, close = false)
        end

        setcolor("gray30")
        setfont("Helvetica", 18)
        text("offset (units)", Point(w / 2, h - 25), halign = :center)
        text("total log-likelihood", Point(ml, mt - 25), halign = :left)

        for (k, (_, label)) in enumerate(curves)
            setcolor(k == 1 ? "black" : "gray45")
            text(label, Point(X(x0) + 20 + 260 * (k - 1), Y(y1) - 20))
        end

        finish()
        return png_view(path)
    end

    p1 = plot_sweep(
        offsets,
        [(curve_lin, "linear, sigma = 0.1"), (curve_log, "log, sigma_y = 0.4")],
        joinpath(OUT_DIR, "Sweep Line.png"))
end

# ╔═╡ 20000000-0000-0000-0000-000000000012
md"""
## Test 2: map over the frame

The swept object is moved over a grid of positions while the other objects stay fixed, and the log-likelihood is mapped. Flat regions are positions the observation cannot distinguish; dark bands are the annuli where the disc crosses field boundaries.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000013
begin
    grid_n = 64
    grid_x = range(-ARENA / 2, ARENA / 2, length = grid_n)
    grid_y = range(-ARENA / 2, ARENA / 2, length = grid_n)

    map_lin = zeros(grid_n, grid_n)
    map_log = zeros(grid_n, grid_n)

    for (iy, y) in enumerate(grid_y), (ix, x) in enumerate(grid_x)
        moved = copy(truth)
        moved[sweep_j] = (pos = (Float64(x), Float64(y)), radius = OBJ_RADIUS)
        pred = means_for(moved, centers, radii)
        map_lin[iy, ix] = ll_linear(obs_means, pred)
        map_log[iy, ix] = ll_log(obs_means, pred)
    end

    floor_lin = maximum(map_lin) - 1.0
    floor_log = maximum(map_log) - 1.0

    println("fraction within 1 nat, linear: ",
            round(count(map_lin .>= floor_lin) / length(map_lin), digits = 3))
    println("fraction within 1 nat, log:    ",
            round(count(map_log .>= floor_log) / length(map_log), digits = 3))
end

# ╔═╡ 20000000-0000-0000-0000-000000000014
begin
    function plot_map(grid, path)
        n = size(grid, 1)
        cell = 8
        Drawing(n * cell + 120, n * cell + 90, path)
        background("white")

        lo = minimum(grid)
        hi = maximum(grid)

        for iy in 1:n, ix in 1:n
            t = hi - lo < 1e-9 ? 0.0 : (grid[iy, ix] - lo) / (hi - lo)
            setcolor(RGB(1.0 - t, 1.0 - t, 1.0 - t))
            x = 60 + (ix - 1) * cell
            y = 40 + (n - iy) * cell
            box(Point(x, y), cell, cell, :fill)
        end

        setcolor("black")
        setline(1)
        rect(Point(60, 40), n * cell, n * cell, :stroke)

        setcolor("gray20")
        setfont("Helvetica", 16)
        text(@sprintf("log-likelihood, range %.0f to %.0f nats", lo, hi),
             Point(60, 20), halign = :left)

        finish()
        return png_view(path)
    end

    p2 = plot_map(map_log, joinpath(OUT_DIR, "Log Likelihood Map.png"))
end

# ╔═╡ 20000000-0000-0000-0000-000000000015
md"""
## Reading

Two numbers decide the question.

- The 1-nat width of the line sweep, compared with the object diameter $2r$. If it exceeds several diameters, the observation does not localise the object and a motion prior with jitter governs its position.
- The fraction of the frame within 1 nat in the map. If it is large, the same conclusion holds in two dimensions.

If both come out flat, the log readout cannot repair it: the flatness comes from the coverage geometry, not from the noise scale.
"""

# ╔═╡ Cell order:
# ╟─20000000-0000-0000-0000-000000000001
# ╟─20000000-0000-0000-0000-000000000002
# ╠═20000000-0000-0000-0000-000000000003
# ╟─20000000-0000-0000-0000-000000000004
# ╟─20000000-0000-0000-0000-000000000005
# ╟─20000000-0000-0000-0000-000000000006
# ╟─20000000-0000-0000-0000-000000000007
# ╟─20000000-0000-0000-0000-000000000008
# ╟─20000000-0000-0000-0000-000000000009
# ╟─20000000-0000-0000-0000-000000000010
# ╟─20000000-0000-0000-0000-000000000011
# ╟─20000000-0000-0000-0000-000000000012
# ╟─20000000-0000-0000-0000-000000000013
# ╟─20000000-0000-0000-0000-000000000014
# ╟─20000000-0000-0000-0000-000000000015
