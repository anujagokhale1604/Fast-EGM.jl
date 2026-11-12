# FastEGM.jl

[![CI](https://github.com/anujagokhale1604/Fast-EGM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anujagokhale1604/Fast-EGM.jl/actions/workflows/CI.yml)
[![Julia 1.11](https://img.shields.io/badge/julia-1.11-blue.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, **zero-allocation** implementation of the Endogenous Grid Method (Carroll, 2006) and Young (2010) Non-Stochastic Lottery Method for solving heterogeneous-agent consumption-saving models with borrowing constraints.

Designed as the steady-state household block for macroeconometric simulation and Sequence-Space Jacobian (SSJ) estimation where policy functions and wealth distributions must be evaluated repeatedly without invoking the garbage collector.

---

## Performance Benchmarks

Measured on Apple Silicon via `BenchmarkTools.jl` (median runtime across 10,000 evaluations):

| Asset Grid Size ($N_a$) | Income States ($N_y$) | Median EGM Step Time | Heap Allocations | Total Memory |
|:-----------------------:|:---------------------:|:--------------------:|:----------------:|:------------:|
| **100**                 | 3                     | **9.75 μs**          | **0**            | **0 bytes**  |
| **300**                 | 3                     | **35.83 μs**         | **0**            | **0 bytes**  |
| **1,000**               | 3                     | **126.29 μs**        | **0**            | **0 bytes**  |

---

## Architectural Principles

1. **Pre-allocated Scratchpad Memory:** Policy updates (`EGMWorkspace`) and Kolmogorov Forward histogram steps (`DistWorkspace`) operate strictly in-place, eliminating runtime heap allocations during steady-state convergence.
2. **Strict Type Stability:** Parameterized structs prevent dynamic dispatch and boxing overhead across inner loops.
3. **Euler Equation Accuracy:** Verified against continuous Euler residual tests away from the borrowing constraint:
   $$\max_{a > \underline{a}, y} \left| 1 - \frac{c_{\text{implied}}}{c} \right| < 2.04 \times 10^{-5}$$
4. **Mass-Conserving Ergodic Distribution:** Uses Young's (2010) non-stochastic lottery weights to map continuous savings policies $a'(a, y)$ onto discrete grid brackets without numerical diffusion, preserving aggregate probability mass ($\sum D = 1$) and mean asset clearing ($\mathbb{E}[a'] = \mathbb{E}[a]$) to machine precision ($< 10^{-11}$).

---

## Quickstart

```julia
using FastEGM

# 1. Initialize model parameters and workspaces
p = ModelParams(Na=300, Ny=3)
ws = EGMWorkspace(p.Na, p.Ny)
c = [p.R * p.a_grid[ia] + p.y_grid[iy] for ia in 1:p.Na, iy in 1:p.Ny]

# 2. Solve steady-state consumption and savings policies in-place
egm_iters = solve_steady_state!(c, ws, p)

# 3. Compute stationary wealth distribution D(a, y) via Young (2010) lottery method
D = zeros(p.Na, p.Ny)
dws = DistWorkspace(p.Na, p.Ny)
dist_iters = solve_stationary_dist!(D, dws, ws.a_new, p)

# 4. Compute aggregate steady-state assets
Ea = aggregate_assets(D, p)