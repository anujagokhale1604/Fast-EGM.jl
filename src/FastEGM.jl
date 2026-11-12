module FastEGM

using BenchmarkTools

export ModelParams, EGMWorkspace, egm_step!, solve_steady_state!, evaluate_euler_errors
export DistWorkspace, lottery_weights!, forward_policy_step!, solve_stationary_dist!, aggregate_assets

# Parametric struct for strict type stability
struct ModelParams{T<:Float64}
    β::T
    γ::T
    R::T
    w::T
    a_min::T
    Na::Int
    Ny::Int
    a_grid::Vector{T}
    y_grid::Vector{T}
    P::Matrix{T}
end

function ModelParams(; β=0.96, γ=2.0, R=1.02, w=1.0, a_min=0.0, Na=300, Ny=3)
    # Exponential asset grid for clustering near the constraint
    a_grid = exp.(range(log(1.0), log(100.0), length=Na)) .- 1.0 .+ a_min
    y_grid = [0.5, 1.0, 1.5]
    P = [0.85 0.10 0.05;
         0.10 0.80 0.10;
         0.05 0.10 0.85]
    return ModelParams(β, γ, R, w, a_min, Na, Ny, a_grid, y_grid, P)
end

# In-place pre-allocated workspace: avoids runtime heap allocations
struct EGMWorkspace{T<:Float64}
    c_new::Matrix{T}
    a_new::Matrix{T}
    c_endo::Matrix{T}
    a_endo::Matrix{T}
    EMU::Matrix{T}
    function EGMWorkspace(Na::Int, Ny::Int)
        new{Float64}(zeros(Na, Ny), zeros(Na, Ny), zeros(Na, Ny), zeros(Na, Ny), zeros(Na, Ny))
    end
end

# Zero-allocation mutating EGM step
function egm_step!(ws::EGMWorkspace, c_prev::Matrix{Float64}, p::ModelParams)
    # 1. Expected Marginal Utility: EMU = P * u'(c_next)
    for iy in 1:p.Ny
        for ia in 1:p.Na
            emu = 0.0
            @inbounds for jy in 1:p.Ny
                c_next = c_prev[ia, jy]
                emu += p.P[iy, jy] * (c_next^(-p.γ))
            end
            ws.EMU[ia, iy] = emu
        end
    end

    # 2. Invert Euler Equation & calculate endogenous asset points
    inv_γ = 1.0 / p.γ
    @inbounds for iy in 1:p.Ny
        y = p.y_grid[iy] * p.w
        for ia in 1:p.Na
            c = (p.β * p.R * ws.EMU[ia, iy])^(-inv_γ)
            ws.c_endo[ia, iy] = c
            ws.a_endo[ia, iy] = (c + p.a_grid[ia] - y) / p.R
        end

        # 3. Linearly interpolate policy onto regular grid
        for ia in 1:p.Na
            cur_a = p.a_grid[ia]
            if cur_a <= ws.a_endo[1, iy]
                # Borrowing constraint binds
                ws.a_new[ia, iy] = p.a_min
                ws.c_new[ia, iy] = cur_a * p.R + y - p.a_min
            else
                k = searchsortedlast(@view(ws.a_endo[:, iy]), cur_a)
                k = min(max(k, 1), p.Na - 1)
                x0, x1 = ws.a_endo[k, iy], ws.a_endo[k+1, iy]
                y0, y1 = p.a_grid[k], p.a_grid[k+1]
                
                slope = (y1 - y0) / (x1 - x0)
                a_prime = y0 + slope * (cur_a - x0)
                
                ws.a_new[ia, iy] = a_prime
                ws.c_new[ia, iy] = cur_a * p.R + y - a_prime
            end
        end
    end
    return nothing
end

# Steady-state solver loop
function solve_steady_state!(c_out::Matrix{Float64}, ws::EGMWorkspace, p::ModelParams; tol=1e-10, max_iter=2000)
    for iter in 1:max_iter
        egm_step!(ws, c_out, p)
        diff = maximum(abs.(ws.c_new .- c_out))
        c_out .= ws.c_new
        if diff < tol
            return iter
        end
    end
    return max_iter
end

# Compute relative Euler residual errors away from the borrowing constraint
function evaluate_euler_errors(c_pol::Matrix{Float64}, a_pol::Matrix{Float64}, p::ModelParams)
    max_error = -Inf
    inv_γ = 1.0 / p.γ
    for iy in 1:p.Ny, ia in 1:p.Na
        if a_pol[ia, iy] <= p.a_min + 1e-4
            continue
        end
        cur_c = c_pol[ia, iy]
        
        rhs_emu = 0.0
        a_prime = a_pol[ia, iy]
        for jy in 1:p.Ny
            k = searchsortedlast(p.a_grid, a_prime)
            k = min(max(k, 1), p.Na - 1)
            w = (a_prime - p.a_grid[k]) / (p.a_grid[k+1] - p.a_grid[k])
            c_tomorrow = c_pol[k, jy] + w * (c_pol[k+1, jy] - c_pol[k, jy])
            rhs_emu += p.P[iy, jy] * (c_tomorrow^(-p.γ))
        end
        c_implied = (p.β * p.R * rhs_emu)^(-inv_γ)
        rel_error = abs(1.0 - c_implied / cur_c)
        max_error = max(max_error, rel_error)
    end
    return max_error
end

# ==============================================================================
# Stationary Distribution Solver (Young, 2010 Non-Stochastic Lottery Method)
# ==============================================================================

# Pre-allocated workspace for zero-allocation Kolmogorov Forward iteration
struct DistWorkspace{T<:Float64}
    idx::Matrix{Int}
    w_low::Matrix{T}
    D_next::Matrix{T}
    function DistWorkspace(Na::Int, Ny::Int)
        new{Float64}(ones(Int, Na, Ny), zeros(Na, Ny), zeros(Na, Ny))
    end
end

# Compute mass-conserving lottery indices and weights from optimal savings policy `a_pol`
function lottery_weights!(dws::DistWorkspace, a_pol::Matrix{Float64}, p::ModelParams)
    @inbounds for iy in 1:p.Ny
        for ia in 1:p.Na
            a_prime = a_pol[ia, iy]
            if a_prime <= p.a_grid[1]
                dws.idx[ia, iy] = 1
                dws.w_low[ia, iy] = 1.0
            elseif a_prime >= p.a_grid[p.Na]
                dws.idx[ia, iy] = p.Na - 1
                dws.w_low[ia, iy] = 0.0
            else
                k = searchsortedlast(p.a_grid, a_prime)
                k = min(max(k, 1), p.Na - 1)
                dws.idx[ia, iy] = k
                dws.w_low[ia, iy] = (p.a_grid[k+1] - a_prime) / (p.a_grid[k+1] - p.a_grid[k])
            end
        end
    end
    return nothing
end

# Single zero-allocation Kolmogorov Forward step: D_next = Lambda' * D_curr
function forward_policy_step!(D_next::Matrix{Float64}, D_curr::Matrix{Float64}, dws::DistWorkspace, p::ModelParams)
    fill!(D_next, 0.0)
    @inbounds for iy in 1:p.Ny
        for ia in 1:p.Na
            mass = D_curr[ia, iy]
            if mass == 0.0
                continue
            end

            k = dws.idx[ia, iy]
            wl = dws.w_low[ia, iy]
            wh = 1.0 - wl

            for jy in 1:p.Ny
                prob = mass * p.P[iy, jy]
                D_next[k, jy]   += wl * prob
                D_next[k+1, jy] += wh * prob
            end
        end
    end
    return nothing
end

# Solve for stationary distribution D(a, y) via power iteration
function solve_stationary_dist!(D_out::Matrix{Float64}, dws::DistWorkspace, a_pol::Matrix{Float64}, p::ModelParams; tol=1e-13, max_iter=10_000)
    fill!(D_out, 1.0 / (p.Na * p.Ny))
    lottery_weights!(dws, a_pol, p)

    for iter in 1:max_iter
        forward_policy_step!(dws.D_next, D_out, dws, p)

        diff = 0.0
        @inbounds for idx in eachindex(D_out)
            d = abs(dws.D_next[idx] - D_out[idx])
            if d > diff
                diff = d
            end
            D_out[idx] = dws.D_next[idx]
        end

        if diff < tol
            return iter
        end
    end
    return max_iter
end

# Aggregate asset holdings E[a] = sum_{i,j} D(i,j) * a_grid[i]
function aggregate_assets(D::Matrix{Float64}, p::ModelParams)
    total = 0.0
    @inbounds for iy in 1:p.Ny
        for ia in 1:p.Na
            total += D[ia, iy] * p.a_grid[ia]
        end
    end
    return total
end

end # module