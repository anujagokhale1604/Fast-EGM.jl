# FastEGM.jl

[![CI](https://github.com/anujagokhale1604/Fast-EGM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anujagokhale1604/Fast-EGM.jl/actions/workflows/CI.yml)
[![Julia 1.11](https://img.shields.io/badge/julia-1.11-blue.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, **zero-allocation** implementation of the Endogenous Grid Method (Carroll, 2006) for solving heterogeneous-agent consumption-saving problems with borrowing constraints.

Designed for macroeconometric simulation and Sequence-Space Jacobian estimation where policy functions must be re-solved repeatedly across parameter grids without invoking the garbage collector.

---

## Performance Benchmarks

Measured on Apple Silicon via `BenchmarkTools.jl` (median runtime across 10,000 evaluations):

| Asset Grid Size ($N_a$) | Income States ($N_y$) | Median Step Time | Heap Allocations | Total Memory |
|:-----------------------:|:---------------------:|:----------------:|:----------------:|:------------:|
| **100**                 | 3                     | **9.75 μs**      | **0**            | **0 bytes**  |
| **300**                 | 3                     | **35.83 μs**     | **0**            | **0 bytes**  |
| **1,000**               | 3                     | **126.29 μs**    | **0**            | **0 bytes**  |

---

## Architectural Principles

1. **Pre-allocated Scratchpad Memory:** State policies and endogenous asset grids update entirely in-place via `EGMWorkspace` structs, eliminating runtime heap allocations during policy updates.
2. **Strict Type Stability:** Fully parameterized structs avoid dynamic dispatch and boxing overhead.
3. **Euler Equation Accuracy:** Validated against continuous Euler residual tests away from the borrowing constraint:
   $$\max_{a > \underline{a}, y} \left| 1 - \frac{c_{\text{implied}}}{c} \right| < 2.04 \times 10^{-5}$$

---

## Quickstart

```julia
using FastEGM

p = ModelParams(Na=300, Ny=3)
ws = EGMWorkspace(p.Na, p.Ny)
c = zeros(p.Na, p.Ny)

# Solve steady-state policy functions in-place
iters = solve_steady_state!(c, ws, p)