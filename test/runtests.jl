using Test
using BenchmarkTools

include("../src/FastEGM.jl")
using .FastEGM

@testset "FastEGM Package Tests" begin
    p = ModelParams(Na=250, Ny=3)
    ws = EGMWorkspace(p.Na, p.Ny)
    c_current = zeros(p.Na, p.Ny)

    # Initial consumption guess: consume return + labor
    for iy in 1:p.Ny, ia in 1:p.Na
        c_current[ia, iy] = p.R * p.a_grid[ia] + p.y_grid[iy]
    end

    # Test 1: Convergence
    iters = solve_steady_state!(c_current, ws, p)
    @test iters < 1500

    # Test 2: Economic properties (monotonicity)
    for iy in 1:p.Ny
        @test issorted(@view(c_current[:, iy]))
        @test issorted(@view(ws.a_new[:, iy]))
    end

    # Test 3: Bounded Euler Residuals
    euler_max = evaluate_euler_errors(c_current, ws.a_new, p)
    println("Maximum Euler Equation Error: ", euler_max)
    @test euler_max < 1e-4

    # Test 4: Memory allocations (0 allocations in inner loop)
    egm_step!(ws, c_current, p) # warm up
    trial = @benchmark egm_step!($ws, $c_current, $p) samples=500 evals=1
    println("Allocated memory per step: ", trial.allocs, " allocations (", trial.memory, " bytes)")
    @test trial.allocs == 0
end