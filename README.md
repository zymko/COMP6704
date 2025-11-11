# Portfolio Optimization and Convergence Benchmark

This repository contains Python implementations for quadratic portfolio optimization under linear constraints and benchmarking of different solvers, including projected gradient descent (PGD), SLSQP, SQP, dual method, and CVXPY.

## Repository Structure

- `complexity_benchmark.py`  
  Benchmarks runtime and accuracy of different quadratic solvers for increasing problem dimensions.  
  Supports PGD, SLSQP, SQP, dual analytic method, and CVXPY.

- `portofilio.py`  
  Implements constrained PGD for quadratic optimization problems:  
  - Constraints: \(x \ge 0\), \(\sum x \le B\), \(\mu^T x \ge R\)  
  - Tracks iteration trajectory for convergence analysis  
  - Contains auxiliary functions for projections onto simplices and halfspaces.

- `plot_SQP_convergence.py`  
  Uses SLSQP to solve quadratic problems of varying sizes and plots convergence curves relative to an analytic QP solution.  

## Usage

### Benchmark solvers
python complexity_benchmark.py
Generates CSV results and runtime plots (complexity_plot.svg, convergence_rates.svg).

### Run PGD portfolio optimization
python portofilio.py
Outputs PGD solution, objective value, expected return, and timing breakdown. Generates pgd_timing_breakdown.png/.pdf.

### Plot SQP convergence
python plot_SQP_convergence.py
Generates per-iteration convergence plots (sqp_convergence_n*.pdf) for different problem dimensions.

## Notes

-Random SPD matrices are generated for testing purposes.

-PGD uses Dykstra’s algorithm for projecting onto the intersection of convex constraints.

-All solvers are compared against the SLSQP quadratic solution for reference.

## License

This project is provided for research and educational purposes.

## Requirements

```bash
python >= 3.10
numpy
scipy
matplotlib
cvxpy
pandas
