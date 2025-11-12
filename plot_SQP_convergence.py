import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def solve_qp_slsqp(Q, mu, budget, target):
    """Solve min x^T Q x s.t. x >= 0, 1^T x <= budget, mu^T x >= target using SLSQP.

    Returns numpy array x*. Raises RuntimeError if SciPy unavailable or solver fails.
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
    res = minimize(fun, x0, method="SLSQP", jac=jac, bounds=bounds, constraints=cons,
                   options={"maxiter": 5000, "ftol": 1e-12, "disp": False})
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    return np.asarray(res.x, dtype=float)

def solve_sqp(Q, mu, budget, target):
    """
    Solve min x^T Q x s.t. x >= 0, sum(x) <= budget, mu^T x >= target using SciPy's SLSQP.
    Tracks iteration trajectory for convergence curve.
    """
    Q = np.asarray(Q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = Q.shape[0]

    def fun(x):
        return float(x @ Q @ x)

    def jac(x):
        return (2.0 * (Q @ x)).astype(float)

    cons = (
        {"type": "ineq", "fun": lambda x: float(budget - np.sum(x)), "jac": lambda x: (-np.ones(n))},
        {"type": "ineq", "fun": lambda x: float(mu @ x - target), "jac": lambda x: mu},
    )
    bounds = [(0.0, None) for _ in range(n)]
    x0 = np.full(n, budget / n, dtype=float)

    traj = []
    def callback(xk):
        traj.append(np.copy(xk))

    res = minimize(
        fun, x0, method="SLSQP", jac=jac,
        bounds=bounds, constraints=cons,
        callback=callback,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False}
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    
    traj.append(res.x.copy())
    return np.asarray(res.x, dtype=float), traj

def plot_sqp_convergence():
    budget = 1.0
    target = 0.12
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for idx, n in enumerate(sizes):
        Q = generate_spd_matrix(n, seed=idx)
        mu = [0.2 + 0.05 * (i / n) for i in range(n)]

        Q_np = np.array(Q, dtype=float)
        mu_np = np.array(mu, dtype=float)

        x_qp = solve_qp_slsqp(Q_np, mu_np, budget, target)      
        x_sqp, traj_sqp = solve_sqp(Q, mu, budget, target)

        if len(traj_sqp) > 1:
            dists_sqp = [float(np.linalg.norm(np.asarray(xk) - x_qp)) for xk in traj_sqp]
            plt.figure(figsize=(6, 4))
            plt.semilogy(dists_sqp, label="||x_k - x_qp||_2 (SQP)", color="orange")
            plt.xlabel("Iteration")
            plt.ylabel("Distance to QP")
            plt.title(f"SQP convergence (n={n})")
            plt.grid(True, which="both", ls=":", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"sqp_convergence_n{n}.pdf", bbox_inches='tight', facecolor='white')
            plt.close()

if __name__ == "__main__":
    plot_sqp_convergence()