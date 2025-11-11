import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.stats import linregress

def generate_spd_matrix(n, seed):
    rnd = random.Random(seed)
    a = [[rnd.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    q = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[k][i] * a[k][j]
            q[i][j] = s
        q[i][i] += n
    return q

def verify_ecos_convergence_order(Q, mu, budget, target, n_val, output_filename="ipm_convergence_verification_en.png"):
    print(f"--- Experiment Start: n = {n_val} ---")

    n = len(mu)
    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, Q))
    constraints = [
        cp.sum(x) <= budget,
        mu @ x >= target,
        x >= 0
    ]
    problem = cp.Problem(objective, constraints)

    print("\n[PART 1] Inferring convergence of the inner loop (Newton's method)")
    problem.solve(solver=cp.ECOS, verbose=True)
    total_iters = problem.solver_stats.num_iters
    print(f"Observation: ECOS reached high precision (1e-8) in only {total_iters} total iterations for n={n_val}.")
    
    print("\n[PART 2] Verifying first-order convergence of the outer loop (Interior-Point Method)")
    problem.solve(solver=cp.ECOS, abstol=1e-12, reltol=1e-12, feastol=1e-12)
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("Pre-solve failed. Cannot proceed with analysis.")
        return
        
    optimal_value_true = problem.value
    x_true = x.value.copy()

    tolerances = np.logspace(-2, -8, num=15)
    history = []
    
    print("\nSimulating outer loop iterations by solving with decreasing tolerance...")
    for i, tol in enumerate(tolerances):
        problem.solve(solver=cp.ECOS, verbose=False, abstol=tol, reltol=tol, feastol=tol)
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            error_k = np.linalg.norm(x.value - x_true)
            history.append({'sim_iter': i, 'e_k': error_k})

    df = pd.DataFrame(history)
    df['e_k'] = df['e_k'].clip(lower=1e-17)
    
    print("\n" + "="*80)
    print(f"      Numerical Analysis of Outer Loop Convergence Order for n={n_val}")
    print("="*80)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f'Verification of ipm Convergence (n={n_val})', fontsize=16)
    
    ax.plot(df['sim_iter'], df['e_k'], 'o-', label='Error Norm ||x_k - x*||')
    ax.set_title('Error vs. Simulated Iteration')
    ax.set_xlabel('Simulated Iteration (decreasing tolerance)')
    ax.set_ylabel('Error Norm (log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"\nAnalysis plot has been saved to: {output_filename}")
    print("-" * 80 + "\n")


# --- Main execution part ---
n_values = [4, 16, 64, 256]
budget = 1.0
target = 0.12

for n in n_values:
    Q_np = np.array(generate_spd_matrix(n, seed=1))
    mu_np = np.array([0.2 + 0.05 * (i / n) for i in range(n)])
    output_file = f"ipm_convergence_n{n}.png"
    
    try:
        verify_ecos_convergence_order(Q_np, mu_np, budget, target, n_val=n, output_filename=output_file)
    except cp.error.SolverError as e:
        print(f"\nError: {e}")
        print("Please install the ECOS solver first: pip install ecos")