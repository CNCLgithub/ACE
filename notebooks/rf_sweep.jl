### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ a0000000-0000-0000-0000-000000000002
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))

    using ACE
    using Luxor
    using Printf
    using Base64: base64encode
    using Markdown: Markdown
end

# ╔═╡ a0000000-0000-0000-0000-000000000001
md"""
# Sweep one object from 3 o'clock to 9 o'clock

**Question.** How much log-likelihood does the model lose when an object's predicted position is on the wrong side of the display?

**Procedure.** Fixation is centred. One object of radius 15 px is placed at the centre of a large peripheral field at 3 o'clock. The observation is the noise-free coverage that this object produces. The predicted position is then translated along the straight line through the fixation point to the 9 o'clock field, and the total log-likelihood is plotted against displacement.

Two densities are scored on the same observation: the linear Gaussian on coverage, $\ell=-\sum (x-\mu)^2/2\sigma^2$ with $\sigma=0.1$, and the log readout currently in the model, $\ell=-\sum [\log(x+\beta)-\log(\mu+\beta)]^2/2\sigma_y^2$ with $\beta=10^{-3}$, $\sigma_y=0.4$. All three colour channels are identical because the object is white on black, so every sum is multiplied by three.
"""

# ╔═╡ a0000000-0000-0000-0000-000000000003
begin
    const ARENA = 400.0
    const R_OBJ = 15.0
    const SIGMA = 0.1
    const SIGMA_Y = 0.4
    const BETA = 1e-3
    const NCH = 3
end

# ╔═╡ a0000000-0000-0000-0000-000000000004
begin
    geom = ACE.generate_receptive_fields((Int(ARENA), Int(ARENA)))
    centers = [(Float64(f.x), Float64(f.y)) for f in geom.fields]
    radii = [Float64(f.r) for f in geom.fields]
    nfields = length(centers)

    function circle_overlap(R, r, d)
        if d >= R + r
            return 0.0
        end
        if d <= abs(R - r)
            return Float64(pi) * min(R, r)^2
        end
        a1 = r^2 * acos((d^2 + r^2 - R^2) / (2d * r))
        a2 = R^2 * acos((d^2 + R^2 - r^2) / (2d * R))
        s = (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
        return a1 + a2 - 0.5 * sqrt(max(s, 0.0))
    end

    function coverage(objects)
        mu = zeros(nfields)
        for (k, c) in enumerate(centers)
            area = 0.0
            for o in objects
                area += circle_overlap(radii[k], o[3], hypot(o[1] - c[1], o[2] - c[2]))
            end
            mu[k] = min(area / (pi * radii[k]^2), 1.0)
        end
        mu
    end

    ll_linear(obs, pred) = -NCH * sum(0.5 * ((o - p) / SIGMA)^2 for (o, p) in zip(obs, pred))

    function ll_log(obs, pred)
        total = 0.0
        for (o, p) in zip(obs, pred)
            x = max(o, -BETA + 1e-9)
            r = log(x + BETA) - log(p + BETA)
            total -= NCH * (0.5 * (r / SIGMA_Y)^2 + log(sqrt(2pi) * SIGMA_Y) + log(x + BETA))
        end
        total
    end

    println("fields: ", nfields, "  radius range: ", round(minimum(radii), digits = 1),
            " to ", round(maximum(radii), digits = 1))
end

# ╔═╡ a0000000-0000-0000-0000-000000000005
md"""
## The object and its observation

The field is chosen as the largest one whose centre lies at 3 o'clock and inside the 400 px frame. Its centre distance $d$ and radius $\rho$ are printed with the coverage the object produces in it. Only one field sees the object, because peripheral centres are further apart than a field radius.
"""

# ╔═╡ a0000000-0000-0000-0000-000000000006
begin
    onscreen = [k for k in eachindex(centers) if hypot(centers[k][1], centers[k][2]) <= ARENA / 2]
    at3 = [k for k in onscreen if abs(atan(centers[k][2], centers[k][1])) < 1e-6]
    k0 = at3[argmax(radii[at3])]

    d0 = hypot(centers[k0][1], centers[k0][2])
    rho0 = radii[k0]
    true_pos = (centers[k0][1], centers[k0][2], R_OBJ)
    obs = coverage([true_pos])

    seen = findall(m -> m > 0, obs)
    @printf("field %d: d = %.1f px, rho = %.1f px\n", k0, d0, rho0)
    @printf("coverage in that field: mu = %.5f, i.e. (r/rho)^2 = %.5f\n", obs[k0], (R_OBJ / rho0)^2)
    println("fields whose coverage is nonzero: ", length(seen), " of ", nfields)
    println("full 9 o'clock displacement: ", round(2d0, digits = 1), " px")
end

# ╔═╡ a0000000-0000-0000-0000-000000000007
md"""
## The sweep

The predicted object moves from the true position through the fixation point to the reflected position. Displacement $s=0$ is the true position, $s=d$ is the fixation point, $s=2d$ is 9 o'clock. The quantity plotted is $\Delta\ell = \ell(s) - \ell(0)$, so every curve starts at zero.
"""

# ╔═╡ a0000000-0000-0000-0000-000000000008
begin
    u = (-centers[k0][1] / d0, -centers[k0][2] / d0)
    ss = range(0.0, 2d0, length = 401)

    base_lin = ll_linear(obs, obs)
    base_log = ll_log(obs, obs)

    dl_lin = Float64[]
    dl_log = Float64[]
    for s in ss
        pred = coverage([(true_pos[1] + s * u[1], true_pos[2] + s * u[2], R_OBJ)])
        push!(dl_lin, ll_linear(obs, pred) - base_lin)
        push!(dl_log, ll_log(obs, pred) - base_log)
    end

    function first_below(curve, offsets, thr)
        j = findfirst(c -> c < thr, curve)
        return j === nothing ? NaN : offsets[j]
    end

    println("                 linear        log")
    @printf("1 nat at       %6.0f px   %6.0f px\n",
            first_below(dl_lin, ss, -1.0), first_below(dl_log, ss, -1.0))
    midpoint = argmin(abs.(ss .- d0))
    @printf("at fixation    %8.1f    %8.1f\n", dl_lin[midpoint], dl_log[midpoint])
    @printf("at 9 o'clock   %8.3f    %8.1f\n", dl_lin[end], dl_log[end])
end

# ╔═╡ a0000000-0000-0000-0000-000000000009
begin
    function png_view(path)
        data = base64encode(String(read(path)))
        return Markdown.parse("![](data:image/png;base64,$data)")
    end

    function panel(X, Y, ss, curve, title, ylim, colour)
        x0, x1 = first(ss), last(ss)
        y0, y1 = ylim
        yc(v) = clamp(v, y0, y1)

        setcolor("gray85")
        setline(1)
        for gy in range(y0, y1, length = 4)
            line(Point(X(x0), Y(yc(gy))), Point(X(x1), Y(yc(gy))), :stroke)
        end

        setcolor("crimson")
        line(Point(X(x0), Y(yc(-1))), Point(X(x1), Y(yc(-1))), :stroke)

        setcolor(colour)
        setline(2.5)
        poly([Point(X(s), Y(yc(v))) for (s, v) in zip(ss, curve)], :stroke, close = false)

        setcolor("black")
        setline(1)
        line(Point(X(x0), Y(y0)), Point(X(x1), Y(y0)), :stroke)
        line(Point(X(x0), Y(y0)), Point(X(x0), Y(y1)), :stroke)

        setcolor("gray30")
        setfont("Helvetica", 15)
        text(title, Point((X(x0) + X(x1)) / 2, Y(y1) - 24), halign = :center)
        text("displacement (px)", Point((X(x0) + X(x1)) / 2, Y(y0) + 34), halign = :center)
        text("Δℓ (nats)", Point(X(x0) - 40, (Y(y0) + Y(y1)) / 2), halign = :center)
        text("-1 nat", Point(X(x1) - 45, Y(yc(-1)) - 12))
    end

    function plot_sweep(ss, dl_lin, dl_log, path)
        Drawing(1000, 460, path)
        background("white")
        ml, mr, mt, mb = 110.0, 40.0, 70.0, 80.0
        gap = 70.0
        w, h = 1000.0, 460.0
        pw = (w - ml - mr - gap) / 2

        Xa(s) = ml + (s - first(ss)) / (last(ss) - first(ss)) * pw
        Xb(s) = ml + pw + gap + (s - first(ss)) / (last(ss) - first(ss)) * pw
        Ya(v) = h - mb - (clamp(v, -4, 0.3) + 4) / 4.3 * (h - mt - mb)
        Yb(v) = h - mb - (clamp(v, -170, 2) + 170) / 172 * (h - mt - mb)

        panel(Xa, Ya, ss, dl_lin, "linear, σ = 0.1   (peak -932 off scale)", (-4.0, 0.3), "steelblue")
        panel(Xb, Yb, ss, dl_log, "log, σy = 0.4   (peak -5418 off scale)", (-170.0, 2.0), "darkorange")

        finish()
        return png_view(path)
    end

    plot_sweep(ss, dl_lin, dl_log, joinpath(@__DIR__, "Sweep 3 to 9.png"))
end

# ╔═╡ a0000000-0000-0000-0000-000000000010
md"""
## Reading

Away from the fixation point the linear readout returns to its starting value: the object may sit anywhere on the opposite side of the display and cost a fraction of a nat. The log readout does not return to zero, because its residual is $\log(1+\mu/\beta)$ rather than $\mu/\sigma$, independent of how small the coverage is.

The peak at the fixation point is large for both, because a 15 px object covers many small foveal fields at once. That peak is a property of the fovea, not of the peripheral mosaic where the tracked objects are.
"""

# ╔═╡ Cell order:
# ╟─a0000000-0000-0000-0000-000000000001
# ╟─a0000000-0000-0000-0000-000000000002
# ╠═a0000000-0000-0000-0000-000000000003
# ╠═a0000000-0000-0000-0000-000000000004
# ╟─a0000000-0000-0000-0000-000000000005
# ╠═a0000000-0000-0000-0000-000000000006
# ╟─a0000000-0000-0000-0000-000000000007
# ╠═a0000000-0000-0000-0000-000000000008
# ╟─a0000000-0000-0000-0000-000000000009
# ╟─a0000000-0000-0000-0000-000000000010
