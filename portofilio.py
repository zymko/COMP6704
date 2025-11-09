import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# from complexity_benchmark import generate_spd_matrix



def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a 1D vector onto the probability simplex {w >= 0, sum w = 1}."""
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1:
        raise ValueError("project_to_simplex expects a 1D array")
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.where(u - (cssv - 1.0) / (np.arange(n) + 1) > 0)[0]
    if rho_idx.size == 0:
        theta = (cssv[-1] - 1.0) / n
    else:
        k = rho_idx[-1]
        theta = (cssv[k] - 1.0) / (k + 1)
    w = np.maximum(v - theta, 0.0)

    return w

def project_nonneg_sum_leq(x: np.ndarray, budget: float) -> np.ndarray:
    """Projection onto {z >= 0, 1^T z <= budget}.

    If sum(pos) <= budget after clipping negatives, return it.
    Otherwise, project onto simplex {z >= 0, 1^T z = budget} via scaled-simplex projection.
    Correct scaling: P_{BΔ}(z) = B * P_{Δ}(z / B).
    """
    z = np.maximum(x, 0.0)
    total = float(z.sum())
    if total <= budget:
        return z
    # Project z/B onto unit simplex, then scale by budget
    return budget * project_to_simplex(x / budget)

def project_halfspace_geq(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Projection onto half-space {a^T z >= b}.
    If a^T x >= b, return x; else x + ((b - a^T x)/||a||^2) a.
    """
    ax = float(a @ x)
    if ax >= b:
        return x
    norm2 = float(a @ a)
    if norm2 <= 0:
        return x
    return x + ((b - ax) / norm2) * a

def dykstra_projection(x0: np.ndarray, projA, projB, iterations: int = 50) -> np.ndarray:
    """Dykstra's algorithm to project x0 onto intersection of two convex sets.

    projA, projB: callables implementing projections onto A and B.
    """
    p = np.zeros_like(x0)
    q = np.zeros_like(x0)
    x = x0.copy()
    for _ in range(iterations):
        y = projA(x + p)
        p = x + p - y
        x = projB(y + q)
        q = y + q - x
    return x

def pgd_quadratic_constrained(
    Q: np.ndarray,
    mu: np.ndarray,
    budget: float,
    ret_lower: float,
    max_iterations: int = 10000,
    tolerance: float = 1e-8,
    step_size: float | None = None,
    proj_inner_iters: int = 50,
    random_seed: int | None = 42,
):
    """Minimize x^T Q x subject to x >= 0, 1^T x <= budget, mu^T x >= ret_lower
    using PGD with Dykstra projection onto intersection.
    """
    Q = np.asarray(Q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = Q.shape[0]
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    if mu.ndim != 1 or mu.shape[0] != n:
        raise ValueError("mu must be 1D and length n")

    if ret_lower > budget * float(mu.max()) + 1e-12:
        raise ValueError("Infeasible: ret_lower exceeds budget * max(mu)")

    if random_seed is not None:
        np.random.seed(random_seed)
    x = np.ones(n, dtype=float)
    x = x / x.sum() * min(budget, 1.0)  # simple feasible-ish init
    # ensure feasibility: enforce mu^T x >= ret_lower using halfspace projection, then nonneg+budget
    x = project_halfspace_geq(x, mu, ret_lower)
    x = project_nonneg_sum_leq(x, budget)

    if step_size is None:
        L = float(np.linalg.eigvalsh(Q).max())
        if L <= 0:
            L = 1.0
        alpha = 1.0 / (2.0 * L)
    else:
        alpha = float(step_size)

    def proj_intersection(z: np.ndarray) -> np.ndarray:
        return dykstra_projection(
            z,
            projA=lambda v: project_nonneg_sum_leq(v, budget),
            projB=lambda v: project_halfspace_geq(v, mu, ret_lower),
            iterations=proj_inner_iters,
        )
    iters = 0
    for k in range(max_iterations):
        grad = 2.0 * (Q @ x)
        x_next = proj_intersection(x - alpha * grad)
        if np.linalg.norm(x_next - x) < tolerance:
            x = x_next
            break
        x = x_next
        iters += 1
    return x, iters 


def pgd_quadratic_constrained_tracked(
    Q: np.ndarray,
    mu: np.ndarray,
    budget: float,
    ret_lower: float,
    max_iterations: int = 10000,
    tolerance: float = 1e-8,
    step_size: float | None = None,
    proj_inner_iters: int = 50,
    random_seed: int | None = 42,
    return_stats: bool = False,
):
    """Same as pgd_quadratic_constrained but returns the iterate trajectory.

    Returns:
        x: final iterate
        trajectory: list of np.ndarray of x at each iteration (including initial)
    """
    Q = np.asarray(Q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = Q.shape[0]
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    if mu.ndim != 1 or mu.shape[0] != n:
        raise ValueError("mu must be 1D and length n")

    if ret_lower > budget * float(mu.max()) + 1e-12:
        raise ValueError("Infeasible: ret_lower exceeds budget * max(mu)")

    if random_seed is not None:
        np.random.seed(random_seed)
    x = np.ones(n, dtype=float)
    x = x / x.sum() * min(budget, 1.0)
    x = project_halfspace_geq(x, mu, ret_lower)
    x = project_nonneg_sum_leq(x, budget)

    if step_size is None:
        L = float(np.linalg.eigvalsh(Q).max())
        if L <= 0:
            L = 1.0
        alpha = 1.0 / (2.0 * L)
    else:
        alpha = float(step_size)

    traj: list[np.ndarray] = [x.copy()]
    # timing stats (seconds)
    stats: dict[str, float] = {
        "grad": 0.0,
        "dykstra_projection": 0.0,
        "project_nonneg_sum_leq": 0.0,
        "project_halfspace_geq": 0.0,
        "total": 0.0,
    }

    def proj_intersection(z: np.ndarray) -> np.ndarray:
        def projA_wrapped(v: np.ndarray) -> np.ndarray:
            t0 = time.perf_counter()
            out = project_nonneg_sum_leq(v, budget)
            stats["project_nonneg_sum_leq"] += time.perf_counter() - t0
            return out

        def projB_wrapped(v: np.ndarray) -> np.ndarray:
            t0 = time.perf_counter()
            out = project_halfspace_geq(v, mu, ret_lower)
            stats["project_halfspace_geq"] += time.perf_counter() - t0
            return out

        t0 = time.perf_counter()
        out = dykstra_projection(
            z,
            projA=projA_wrapped,
            projB=projB_wrapped,
            iterations=proj_inner_iters,
        )
        stats["dykstra_projection"] += time.perf_counter() - t0
        return out

    t_total_start = time.perf_counter()
    for _ in range(max_iterations):
        t0 = time.perf_counter()
        grad = 2.0 * (Q @ x)
        stats["grad"] += time.perf_counter() - t0
        x_next = proj_intersection(x - alpha * grad)
        traj.append(x_next.copy())
        if np.linalg.norm(x_next - x) < tolerance:
            x = x_next
            break
        x = x_next
    stats["total"] += time.perf_counter() - t_total_start

    if return_stats:
        return x, traj, stats
    return x, traj

if __name__ == "__main__":

    df = pd.DataFrame(
        [
            [93.043, 51.826, 1.063],
            [84.585, 52.823, 0.938],
            [111.453, 56.477, 1.000],
            [99.525, 49.805, 0.938],
            [95.819, 50.287, 1.438],
            [114.708, 51.521, 1.700],
            [111.515, 51.531, 2.540],
            [113.211, 48.664, 2.390],
            [104.942, 55.744, 3.120],
            [99.827, 47.916, 2.980],
            [91.607, 49.438, 1.900],
            [107.937, 51.336, 1.750],
            [115.590, 55.081, 1.800],
        ],
        columns=["IBM", "WMT", "SEHI"],
    )

    returns = df.pct_change().iloc[1:, :]
    r_mean = returns.mean()
    r_cov = returns.cov()

    Sigma = r_cov.values
    mu = r_mean.values
    asset_names = list(r_cov.columns)

    # B = 1000.0
    # R = 50.0

    B = 1.0
    R = 0.12
    n=2048

    Q = generate_spd_matrix(n, seed=10)
    mu = [0.2 + 0.05 * (i / n) for i in range(n)]

    # x_constrained = pgd_quadratic_constrained(
    #     Q=Q,
    #     mu=mu,
    #     budget=B,
    #     ret_lower=R,
    #     max_iterations=20000,
    #     tolerance=1e-12,
    #     proj_inner_iters=100,
    # )

    
    # x_series = pd.Series(x_constrained, index=asset_names, name="x_constrained")
    # obj = float(x_constrained.T @ Sigma @ x_constrained)
    # ret = float(mu @ x_constrained)
    # sum_x = float(x_constrained.sum())

    # print("PGD constrained results")
    # print(f"Objective (x^T Q x): {obj:.6f}")
    # print(f"Sum of weights: {sum_x:.6f}")
    # print(f"Expected return (mu^T x): {ret:.6f}")
    # print("\nWeights:")
    # formatted_series = x_series.map(lambda v: f"{v:.6f}")
    # print(formatted_series.to_string())


    # --- Convergence to analytic solution (per-iteration) ---
    # Analytic reference solution (from QP):
    # x_star = np.array([497.05, 0.0, 502.95], dtype=float)

    # Run tracked PGD to collect iterates and timing stats
    _, trajectory, timing_stats = pgd_quadratic_constrained_tracked(
        Q=Q,
        mu=mu,
        budget=B,
        ret_lower=R,
        max_iterations=20000,
        tolerance=1e-12,
        proj_inner_iters=100,
        return_stats=True,
    )

    # # Compute per-iteration differences
    # diff_norms = [float(np.linalg.norm(xk - x_star)) for xk in trajectory]
    # diff_IBM = [float(xk[0] - x_star[0]) for xk in trajectory]
    # diff_WMT = [float(xk[1] - x_star[1]) for xk in trajectory]
    # diff_SEHI = [float(xk[2] - x_star[2]) for xk in trajectory]

    # # Plot L2 distance (semilogy)
    # plt.figure(figsize=(6, 4))
    # plt.semilogy(diff_norms, label="||x_k - x*||_2")
    # plt.xlabel("Iteration")
    # plt.ylabel("Distance to analytic")
    # plt.title("PGD convergence to analytic solution (L2 norm)")
    # plt.grid(True, which="both", ls=":", alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("portofilio.pdf", bbox_inches='tight', facecolor='white')
    # plt.show()

    # --- Performance profile (timings) ---
    print("\nPGD timing breakdown (seconds):")
    for k in [
        "grad",
        "project_nonneg_sum_leq",
        "project_halfspace_geq",
        "dykstra_projection",
        "total",
    ]:
        print(f"  {k}: {timing_stats.get(k, 0.0):.6f}")

    # Plot timing breakdown
    components = [
        "grad",
        "project_to_simplex",
        "project_halfspace",
        "dykstra_projection",
    ]
    values = [float(timing_stats.get(k, 0.0)) for k in components]
    # log 轴不允许 0，做一个极小值下限以保证可显示
    eps = 1e-12
    labels = components
    times = [max(v, eps) for v in values]

    plt.figure(figsize=(4.8, 3.0))
    bars = plt.bar(labels, times, color=["#cecece", "#a559aa", "#59a89c", "#f0c571"]) 
    plt.yscale("log")
    plt.ylabel("Time (s) [log]", fontsize=9)
    plt.title("PGD timing breakdown", fontsize=11)
    plt.xticks(rotation=15, fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    for b, v in zip(bars, values):
        v_txt = max(v, eps)
        plt.text(b.get_x() + b.get_width() / 2, v_txt, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    # smaller PNG for easy display
    plt.savefig("pgd_timing_breakdown.png", bbox_inches='tight', facecolor='white', dpi=120)
    # keep a vector version as well (optional viewing)
    plt.savefig("pgd_timing_breakdown.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
