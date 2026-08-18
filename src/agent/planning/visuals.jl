
function paint_state(m::MentalModule{TargetDesignation},
                     drawing, ret_finish=true)
    paint_rg_predictions!(drawing, m)
    paint_rg_expectation!(drawing, m)
    ret_finish && finish()
    return drawing
end

function paint_rg_predictions!(drawing, m::MentalModule{TargetDesignation})
    p, s = mparse(m)
    opacity = 2.0 / length(s.chain)
    for trace = s.chain
        states = get_states(trace)
        for t = 1:length(states)
            state = states[t]
            # make future more transparent
            o = opacity * exp(-0.02*t)
            for i = 1:ndynamic(state)
                x, pos, _ = state.dynamic[i]
                paint!(x.shape, pos, x.color, o, 0.3)
            end
        end
    end

    return nothing
end
function paint_rg_expectation!(drawing, m::MentalModule{TargetDesignation})
    p, s = mparse(m)
    paint_rg_bar!(s.expectation, drawing.width, drawing.height)
    return nothing
end


function paint_rg_bar!(
    rg::Float64, # ranges from [-1, 1]
    canvas_w::Float64,
    canvas_h::Float64,
    bar_w::Float64  = 16.0,
    bar_h::Float64  = 200.0,
    margin::Float64 = 20.0,
    hue_low::Float64  = 0.35,       # green  (hue in [0,1])
    hue_high::Float64 = 0.0,        # red
)
    rg = clamp(rg, -1.0, 1.0)
    rg_scaled = 0.5 * (rg + 1.0)   # [-1, 1] -> [0, 1] for colour interpolation

    # ── Position: left side of canvas, vertically centred ───────────────────
    # Luxor origin() is at canvas centre, y-axis points DOWN
    bar_x = -canvas_w/2 + margin + bar_w/2
    bar_y = -bar_h/2                # top of bar in Luxor coords

    # ── Background track ─────────────────────────────────────────────────────
    @layer begin
        sethue(0.15, 0.15, 0.15)
        setopacity(0.4)
        Luxor.box(Point(bar_x, 0.0), bar_w, bar_h, :fill)
    end

    # ── Filled portion (grows from middle) ───────────────────────────────────
    # rg > 0  →  fill grows upward   (negative y in Luxor)
    # rg < 0  →  fill grows downward (positive y in Luxor)
    fill_h  = bar_h * abs(rg) / 2       # half-bar = full extent when |rg| == 1
    fill_cy = -(rg / 2) * (bar_h / 2)  # centre of filled rect, in Luxor coords
                                        # rg>0: negative y (upward); rg<0: positive y (downward)

    hue = hue_low + rg_scaled * (hue_high - hue_low)
    fill_color = Colors.HSV(hue * 360, 0.85, 0.9)

    @layer begin
        sethue(fill_color)
        setopacity(0.9)
        Luxor.box(Point(bar_x, fill_cy), bar_w, fill_h, :fill)
    end

    # ── Centre tick ───────────────────────────────────────────────────────────
    @layer begin
        sethue(0.8, 0.8, 0.8)
        setopacity(0.9)
        setline(1.0)
        line(Point(bar_x - bar_w/2, 0.0), Point(bar_x + bar_w/2, 0.0), :stroke)
    end

    # ── Border ────────────────────────────────────────────────────────────────
    @layer begin
        sethue(0.7, 0.7, 0.7)
        setopacity(0.8)
        setline(1.0)
        Luxor.box(Point(bar_x, 0.0), bar_w, bar_h, :stroke)
    end

    # ── Label ─────────────────────────────────────────────────────────────────
    @layer begin
        sethue(0.0, 0.0, 0.0)
        setopacity(1.0)
        fontsize(9)
        text("$(round(rg; digits=3))",
             Point(bar_x, bar_y - 8.0),
             halign=:center, valign=:bottom)
        text("Pr(R)-Pr(G)",
             Point(bar_x, bar_h/2 + 12.0),
             halign=:center, valign=:bottom)
    end

    return nothing
end
