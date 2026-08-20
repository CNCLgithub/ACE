using Gen
using ACE

function test_wm()
    istate = WorldState([
        Disc(S2V(-10, 10), S2V(1, 0), 5.0),
        Disc(S2V(10, 10), S2V(-1, 0), 5.0),
        Disc(S2V(0, 10), S2V(0.5, -1), 10.0),
    ])

    motion = BrownianVel()
    graphics = RFGraphics((400, 400), 10.0, 2.0, 0.1, S2V32(0,0), istate)
    wm = WorldModel(motion, graphics)


    trace, ls = generate(time_kernel, (1, istate, wm))
    display(get_choices(trace))
    @show ls
    return nothing
end

test_wm();

function test_perception()
end
