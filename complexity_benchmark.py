import math
import random
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from portofilio import pgd_quadratic_constrained_tracked
import cvxpy as cp
from scipy.optimize import minimize
import csv



def generate_spd_matrix(n, seed):
    rnd = random.Random(seed)
    # Build a random matrix A then return A^T A + n*I to ensure SPD.
    a = [[rnd.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    q = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[k][i] * a[k][j]
            q[i][j] = s
        q[i][i] += n  # add diagonal dominance
    return q


def mat_vec(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def solve_qp_slsqp(Q, mu, budget, target):
    """Solve min x^T Q x s.t. x >= 0, 1^T x <= budget, mu^T x >= target using SLSQP.

    Returns (x*, trajectory). Raises RuntimeError if solver fails.
    """
    Q = np.asarray(Q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = Q.shape[0]

    def fun(x):
        x = np.asarray(x)
        return float(x @ Q @ x)

    def jac(x):
        x = np.asarray(x)
        return (2.0 * (Q @ x)).astype(float)

    cons = (
        {"type": "ineq", "fun": lambda x: float(budget - np.sum(x)), "jac": lambda x: (-np.ones(n))},
        {"type": "ineq", "fun": lambda x: float(mu @ x - target), "jac": lambda x: mu},
    )
    bounds = [(0.0, None) for _ in range(n)]
    x0 = np.full(n, budget / n, dtype=float)
    trajectory = []
    def _cb(xk):
        trajectory.append(np.asarray(xk, dtype=float))

    res = minimize(
        fun, x0, method="SLSQP", jac=jac, bounds=bounds, constraints=cons,
        options={"maxiter": 5000, "ftol": 1e-12, "disp": False}, callback=_cb
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    trajectory.append(np.asarray(res.x, dtype=float))
    return np.asarray(res.x, dtype=float), trajectory



def gaussian_elimination(mat, rhs):
    size = len(rhs)
    augmented = [row[:] + [rhs[idx]] for idx, row in enumerate(mat)]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(augmented[r][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular system")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, size + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][size] for i in range(size)]


def solve_dual(Q, mu, budget, target):
    n = len(Q)
    size = n + 2
    mat = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    for i in range(n):
        for j in range(n):
            mat[i][j] = 2.0 * Q[i][j]
    for i in range(n):
        mat[i][n] = 1.0
        mat[i][n + 1] = mu[i]
        mat[n][i] = 1.0
        mat[n + 1][i] = mu[i]

    rhs[n] = budget
    rhs[n + 1] = target
    solution = gaussian_elimination(mat, rhs)
    x = solution[:n]
    return x

def solve_cvxpy(Q, mu, budget, target):
    n = len(mu)
    x = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(x, Q))
    
    constraints = [
        cp.sum(x) <= budget,
        mu @ x >= target,
        x >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    # problem.solve(solver=cp.ECOS, verbose=False)
    problem.solve(verbose=False)
    
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return x.value
    else:
        return np.zeros(n)


def solve_sqp(Q, mu, budget, target):
    n = len(Q)

    def objective_function(x):
        return x.T @ Q @ x
    def objective_gradient(x):
        return 2 * (Q @ x)

    cons1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - budget}
    cons2 = {'type': 'ineq', 'fun': lambda x: np.dot(mu, x) - target}
    constraints = [cons1, cons2]

    bounds = tuple((0, None) for _ in range(n))

    initial_guess = np.full(n, budget / n)

    trajectory = []
    def _cb(xk):
        # record iterate
        trajectory.append(np.asarray(xk, dtype=float))

    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        jac=objective_gradient,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-7},
        callback=_cb,
    )

    # include final solution as the last iterate for consistency
    trajectory.append(np.asarray(result.x, dtype=float))
    return result.x, trajectory


def benchmark():
    budget = 1.0
    target = 0.12
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    rows = []

    for idx, n in enumerate(sizes):
        Q = generate_spd_matrix(n, seed=idx)
        mu = [0.2 + 0.05 * (i / n) for i in range(n)]

        Q_np = np.array(Q, dtype=float)
        mu_np = np.array(mu, dtype=float)

        # Run PGD (tracked) to get trajectory for convergence analysis
        start = time.perf_counter()
        x_pgd, traj = pgd_quadratic_constrained_tracked(
            Q=Q_np, mu=mu_np, budget=budget, ret_lower=target,
            max_iterations=5000, tolerance=1e-9, proj_inner_iters=50,
        )
        pgd_time = time.perf_counter() - start
        iters = len(traj) - 1

        start = time.perf_counter()
        x_dual = solve_dual(Q, mu, budget, target)
        dual_time = time.perf_counter() - start

        # Solve QP as analytic reference (if available)
        start = time.perf_counter()
        x_qp, traj_qp = solve_qp_slsqp(Q_np, mu_np, budget, target)
        qp_ok = True
        slsqp_time = time.perf_counter() - start

        start = time.perf_counter()
        x_cvxpy = solve_cvxpy(Q_np, mu_np, budget, target)
        cvxpy_time = time.perf_counter() - start  

        start = time.perf_counter()
        x_sqp, traj_sqp = solve_sqp(Q, mu, budget, target)
        sqp_time = time.perf_counter() - start    


        # Accuracy metrics at termination (distance to optimal solution only)
        final_dist = float(np.linalg.norm(np.asarray(x_pgd) - x_qp))

        # Dual/CVXPY/SQP solution errors vs QP
        x_dual_np = np.asarray(x_dual, dtype=float)
        dist_dual = float(np.linalg.norm(x_dual_np - x_qp))
        x_cvxpy_np = np.asarray(x_cvxpy, dtype=float)
        dist_cvxpy = float(np.linalg.norm(x_cvxpy_np - x_qp))
        x_sqp_np = np.asarray(x_sqp, dtype=float)
        dist_sqp = float(np.linalg.norm(x_sqp_np - x_qp))

        # Print concise error summary vs QP solver
        print(
            f"[n={n}] dist_to_SLSQP — PGD={final_dist:.3e}, Dual={dist_dual:.3e}, CVXPY={dist_cvxpy:.3e}, SQP={dist_sqp:.3e}"
        )

        # No combined plot; each method is plotted separately above.

        rows.append(
            {
                "n": n,
                "pgd_time": pgd_time,
                "pgd_iters": iters,
                "dual_time": dual_time,
                "cvxpy_time": cvxpy_time,
                "sqp_time": sqp_time,
                "slsqp_time": slsqp_time,
                "dist_pgd": final_dist,
                "dist_dual": dist_dual,
                "dist_cvxpy": dist_cvxpy,
                "dist_sqp": dist_sqp,
            }
        )

    return rows


def save_results_csv(rows, csv_path):
    fieldnames = [
        "n",
        "pgd_time",
        "pgd_iters",
        "dual_time",
        "cvxpy_time",
        "sqp_time",
        "slsqp_time",
        "dist_pgd",
        "dist_dual",
        "dist_cvxpy",
        "dist_sqp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_time_by_dimension(rows, out_svg):
    rows_sorted = sorted(rows, key=lambda r: r["n"])
    ns = [r["n"] for r in rows_sorted]
    methods = [
        ("PGD (ours)", "pgd_time"),
        ("Dual (analytic)", "dual_time"),
        ("CVXPY (quadprog)", "cvxpy_time"),
        ("SLSQP (SciPy)", "slsqp_time"),
        ("SQP (custom)", "sqp_time"),
    ]

    plt.figure(figsize=(6.5, 4.2))
    markers = ["o", "s", "^", "D", "v"]
    for (label, key), marker in zip(methods, markers):
        ys = [float(r[key]) for r in rows_sorted]
        plt.plot(ns, ys, marker=marker, linewidth=1.8, markersize=6, label=label)

    plt.xticks(ns, [str(n) for n in ns])
    plt.xlabel("Dimension n")
    plt.ylabel("Time (s)")
    plt.title("Runtime vs Dimension by Method")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches='tight', facecolor='white')
    try:
        p = Path(out_svg)
        plt.savefig(str(p.with_suffix('.png')), bbox_inches='tight', facecolor='white', dpi=200)
        plt.savefig(str(p.with_suffix('.pdf')), bbox_inches='tight', facecolor='white')
    except Exception:
        pass
    plt.close()


def plot_iteration_errors(n, curves, out_path):
    # curves: list of (label, dist_list)
    plt.figure(figsize=(6.5, 4.2))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for (label, dists), mk in zip(curves, markers):
        xs = list(range(len(dists)))
        plt.plot(xs, dists, marker=mk, linewidth=1.6, markersize=5, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Error ||x_k - x_qp||_2")
    plt.title(f"Per-iteration error (n={n})")
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    try:
        p = Path(out_path)
        plt.savefig(str(p.with_suffix('.png')), bbox_inches='tight', facecolor='white', dpi=200)
    except Exception:
        pass
    plt.close()

def plot_convergence_rates(rows, out_svg):
    rows_sorted = sorted(rows, key=lambda r: r["n"])
    ns = [r["n"] for r in rows_sorted]

    rho_pgd = [r.get("rho_pgd", float("nan")) for r in rows_sorted]
    rho_slsqp = [r.get("rho_slsqp", float("nan")) for r in rows_sorted]
    alpha_pgd = [r.get("alpha_pgd", float("nan")) for r in rows_sorted]
    alpha_slsqp = [r.get("alpha_slsqp", float("nan")) for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ns, rho_pgd, marker="o", label="PGD ρ")
    ax.plot(ns, rho_slsqp, marker="s", label="SLSQP ρ")
    ax.set_xlabel("Dimension n")
    ax.set_ylabel("Linear rate ρ")
    ax.set_title("Geometric rate vs n")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ns, alpha_pgd, marker="o", label="PGD α")
    ax.plot(ns, alpha_slsqp, marker="s", label="SLSQP α")
    ax.set_xlabel("Dimension n")
    ax.set_ylabel("Sublinear order α")
    ax.set_title("Sublinear order vs n")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches='tight', facecolor='white')
    p = Path(out_svg)
    plt.savefig(str(p.with_suffix('.png')), bbox_inches='tight', facecolor='white', dpi=200)
    plt.savefig(str(p.with_suffix('.pdf')), bbox_inches='tight', facecolor='white')
    plt.close()



def main():
    rows = benchmark()
    out_dir = Path(__file__).parent

    print("n | PGD time (s) | PGD iters | Dual time (s) | CVXPY time (s) | SQP time (s) | SLSQP time (s) | dist(PGD) | dist(Dual) | dist(CVXPY) | dist(SQP)")
    for row in rows:
        print(
            f"{row['n']:2d} | {row['pgd_time']:.6f} | {row['pgd_iters']:9d} | {row['dual_time']:.6f} | {row['cvxpy_time']:.6f} | {row['sqp_time']:.6f} | {row['slsqp_time']:.6f} | "
            f"{row['dist_pgd']:.3e} | {row['dist_dual']:.3e} | {row['dist_cvxpy']:.3e} | {row['dist_sqp']:.3e}"
        )

    # Save CSV and plot comparison across dimensions
    save_results_csv(rows, str(out_dir / "complexity_table.csv"))
    plot_time_by_dimension(rows, str(out_dir / "complexity_plot.svg"))
    plot_convergence_rates(rows, str(out_dir / "convergence_rates.svg"))

if __name__ == "__main__":
    main()
