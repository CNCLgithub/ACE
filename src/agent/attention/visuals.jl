using Luxor: sethue, Point, circle, setopacity

function paint_state(att::MentalModule{AdaptiveComputation},
                     dec::MentalModule{TargetDesignation},
                     drawing, ret_finish=true)
    # paint_attention_hashmap!(drawing, att)
    paint_fixation_prediction!(drawing, att, dec)
    paint_attention_load!(drawing, att)
    ret_finish && finish()
    return drawing
end

function paint_attention_load!(drawing, att::MentalModule{AdaptiveComputation})
    protocol, state = mparse(att)
    avg_load = state.avg_load
    load_percent = avg_load / protocol.load
    paint_attention_bar!(load_percent, drawing.width, drawing.height)
    return nothing
end

function paint_attention_bar!(
    load::Float64,                  # value in [0, 1]
    canvas_w::Float64,
    canvas_h::Float64,
    bar_w::Float64  = 16.0,
    bar_h::Float64  = 200.0,
    margin::Float64 = 20.0,
    hue_low::Float64  = 220.0/355.0,       # green  (hue in [0,1])
    hue_high::Float64 = 0.0,        # red
)
    load = clamp(load, 0.02, 1.0)

    # ── Position: right side of canvas, vertically centred ──────────────────
    # Luxor origin() is at canvas centre, y-axis points DOWN
    bar_x = canvas_w/2 - margin - bar_w/2
    bar_y = -bar_h/2                # top of bar in Luxor coords

    # ── Background track ────────────────────────────────────────────────────
    @layer begin
        sethue(0.15, 0.15, 0.15)   # dark track
        setopacity(0.4)
        Luxor.box(Point(bar_x, 0.0), bar_w, bar_h, :fill)
    end

    # ── Filled portion (grows upward from bottom) ───────────────────────────
    fill_h   = bar_h * load
    fill_top = bar_y + (bar_h - fill_h)          # top edge of filled rect
    fill_cy  = fill_top + fill_h/2               # centre in Luxor coords

    r,g,b = _interpolate_color(S3V(0., 0., 1.), S3V(1., 0., 0.), load)
    # hue  = hue_low + load * (hue_high - hue_low) # interpolate green→red
    # fill_color = Colors.HSV(hue * 360, 0.85, 0.9)

    @layer begin
        sethue(r,g,b)
        setopacity(0.9)
        Luxor.box(Point(bar_x, fill_cy), bar_w, fill_h, :fill)
    end

    # ── Border ───────────────────────────────────────────────────────────────
    @layer begin
        sethue(0.7, 0.7, 0.7)
        setopacity(0.8)
        setline(1.0)
        Luxor.box(Point(bar_x, 0.0), bar_w, bar_h, :stroke)
    end

    # ── Label ────────────────────────────────────────────────────────────────
    @layer begin
        sethue(0.0, 0.0, 0.0)
        setopacity(1.0)
        fontsize(9)
        text("$(round(Int, load*100))%",
             Point(bar_x, bar_y - 8.0),
             halign=:center, valign=:bottom)
        text("Load",
             Point(bar_x, bar_h/2 + 12.0),
             halign=:center, valign=:bottom)
    end

    return nothing
end

function paint_attention_hashmap!(drawing, att::MentalModule{AdaptiveComputation})
    protocol, state = mparse(att)
    # No data
    isempty(state.dPi) && return nothing
    # Percent load
    load_pct = 1.5*state.avg_load / protocol.load
    npoints = length(state.dPi.samples)
    ws = softmax(collect(state.dPi.samples), protocol.itemp)
    lmul!(load_pct / maximum(ws), ws)
    for i = 1:npoints
        x,y,t = state.dPi.coords[i]
        w = ws[i]
        r,g,b = _interpolate_color(S3V(0., 0., 1.), S3V(1., 0., 0.), w)
        radius = clamp(w*15.0, 5.0, 15.0)
        opacity = clamp(w * 0.8, 0.05, 1.0)
        # Luxor commands
        @layer begin
            setopacity(opacity)
            sethue(r,g,b)
            point = Point(x, -y)
            # circle(point, 10.0, :fill)
            star(point, radius, 5, 0.5, 0.0, :fill)
        end
    end
    return nothing
end

function _interpolate_color(a::S3V, b::S3V, w::Float64)
    a + (w .* (b - a))
end

function paint_fixation_prediction!(drawing,
                                    attention::MentalModule{AdaptiveComputation},
                                    decision::MentalModule{TargetDesignation})
    attp, atts = mparse(attention)
    # No data
    isempty(atts.dPi) && return nothing
    # Percent load - modulates fixation spread
    load_pct = atts.avg_load / attp.load

    time_sigma = clamp(1.5 * load_pct, 0.15, 0.90)
    paint_simulation_traverse!(decision, time_sigma)
    return nothing
end

function paint_simulation_traverse!(m::MentalModule{TargetDesignation}, stop_weight::Float64)
    p, s = mparse(m)
    for trace = s.chain
        paint_trace_travese!(trace, stop_weight)
    end
    return nothing
end

function paint_trace_travese!(trace::KMDTrace, stop::Float64,
                              hue_low = .35, hue_high = 0.0)
    rg,_ = get_retval(trace)
    rg = clamp(rg, -1.0, 1.0)
    rg_scaled = 0.5 * (rg + 1.0)
    hue = hue_low + rg_scaled * (hue_high - hue_low)
    fill_color = Colors.HSV(hue * 360, 0.85, 0.9)

    states = get_states(trace)
    nt = length(states)
    nt === 0 && return nothing
    t = nt === 1 ? 1 : binom(nt-1, 1-stop)+1
    state = states[t]
    for i = 1:ndynamic(state)
        x, pos, _ = state.dynamic[i]
        point = Luxor.Point(pos[1], -pos[2])
        @layer begin
            setopacity(1.0)
            sethue(fill_color)
            translate(point)
            Luxor.circle(Luxor.Point(0, 0), 5.0, :fill)
        end
    end
    return nothing
end

# function render_attention(att::MentalModule{UniformProtocol})
#     return nothing
# end
