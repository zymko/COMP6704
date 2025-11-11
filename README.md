# Portfolio Optimization and Convergence Benchmark

This repository contains Python implementations for quadratic portfolio optimization under linear constraints and benchmarking of different solvers, including projected gradient descent (PGD), SLSQP, SQP, dual method, and CVXPY.

The project centers on the Markowitz Portfolio Optimization problem — a classical model in financial decision-making that seeks to minimize portfolio risk (variance) while achieving a target expected return. The optimization problem is quadratic and convex, ensuring global optimality and making it ideal for algorithmic performance comparison across convex solvers.

## Problem Background

Portfolio optimization plays a central role in modern finance, where investors aim to balance expected returns against investment risks. Following Markowitz’s mean-variance framework, risk is quantified by the covariance of asset returns, and the optimization seeks a trade-off between maximizing returns and minimizing volatility.

In this repository, we study an investment scenario with a budget $B$ distributed among three non-dividend-paying stocks — IBM, WMT, and SEHI — over a one-month horizon. The problem setup and dataset originate from a Georgia Tech tutorial by Professor Shabbir Ahmed, based on real market return data. The model assumes a buy-and-hold strategy, where the investor purchases shares at the beginning of the month and sells them at the end.

The mathematical formulation is:

$$
\begin{aligned}
\min_{x} \quad & x^\top Q x \\
\text{s.t.} \quad
& \sum_i x_i \le B, \\
& \bar{\mu}^\top x \ge R, \\
& x \ge 0,
\end{aligned}
$$

where $Q$ is the covariance matrix of asset returns, $\bar{\mu}$ is the expected monthly return vector, and $R$ represents the target return level. This convex quadratic formulation guarantees global optimality under affine constraints.

## Dataset Description

Two types of datasets are used in this project:

- **Real data:** Historical monthly returns of IBM, WMT, and SEHI are used to estimate the expected return vector $\bar{\mu}$ and the covariance matrix $Q$. These values form the basis of the Markowitz portfolio optimization model for a one-month horizon.

- **Synthetic data:** For scalability and complexity analysis, random symmetric positive definite (SPD) matrices are generated to simulate high-dimensional covariance structures with controllable size $n$. This enables systematic benchmarking of solver performance across increasing problem dimensions and constraint counts.

The real dataset provides practical insight into financial optimization under realistic conditions, while the synthetic datasets allow controlled experimentation on solver efficiency, convergence rate, and numerical stability.

## Repository Structure

- `complexity_benchmark.py`  
  Benchmarks runtime and accuracy of different quadratic solvers for increasing problem dimensions.  
  Supports PGD, SLSQP, SQP, dual analytic method, and CVXPY.

- `portofilio.py`  
  Implements constrained PGD for quadratic optimization problems:  
  - Constraints: $x \ge 0$, $\sum x \le B$, $\mu^T x \ge R$  
  - Tracks iteration trajectory for convergence analysis  
  - Contains auxiliary functions for projections onto simplices and halfspaces.

- `plot_SQP_convergence.py`  
  Uses SLSQP to solve quadratic problems of varying sizes and plots convergence curves relative to an analytic QP solution.  

## Usage

### Benchmark solvers
```bash
python complexity_benchmark.py
```
Generates CSV results and runtime plots (complexity_plot.svg, convergence_rates.svg).

### Run PGD portfolio optimization
```bash
python portofilio.py
```
Outputs PGD solution, objective value, expected return, and timing breakdown. Generates pgd_timing_breakdown.png/.pdf.

### Plot SQP convergence
```bash
python plot_SQP_convergence.py
```
Generates per-iteration convergence plots (sqp_convergence_n*.pdf) for different problem dimensions.

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
```