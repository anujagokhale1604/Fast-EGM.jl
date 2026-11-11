using BenchmarkTools

if !@isdefined(FastEGM)
    include(joinpath(@__DIR__, "..", "src", "FastEGM.jl"))
end

println("Running EGM Performance Benchmarks across grid dimensions...")

for Na in [100, 300, 1000]
    p = FastEGM.ModelParams(Na=Na, Ny=3)
    ws = FastEGM.EGMWorkspace(p.Na, p.Ny)
    c = zeros(p.Na, p.Ny)
    FastEGM.solve_steady_state!(c, ws, p)

    b = @benchmark FastEGM.egm_step!($ws, $c, $p)
    med_time = BenchmarkTools.prettytime(median(b).time)
    allocs = b.allocs
    bytes = b.memory

    println("Grid Na = $(rpad(Na, 4)) x Ny = 3 | Median Time: $(rpad(med_time, 9)) | Allocs: $(allocs) ($(bytes) bytes)")
end