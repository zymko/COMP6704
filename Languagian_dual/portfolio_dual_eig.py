import math
from itertools import combinations

PRICE_DATA = [
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
]

ASSET_NAMES = ["IBM", "WMT", "SEHI"]
TARGET_RETURN = 0.05
BUDGET = 1000.0
TOL = 1e-9


def compute_returns(prices):
    return [
        [(curr[i] - prev[i]) / prev[i] for i in range(len(ASSET_NAMES))]
        for prev, curr in zip(prices, prices[1:])
    ]


def column_stats(returns):
    n = len(returns)
    means = []
    for i in range(len(ASSET_NAMES)):
        means.append(sum(row[i] for row in returns) / n)

    cov = []
    for i in range(len(ASSET_NAMES)):
        row = []
        for j in range(len(ASSET_NAMES)):
            total = sum(
                (returns[t][i] - means[i]) * (returns[t][j] - means[j]) for t in range(n)
            )
            row.append(total / (n - 1))
        cov.append(row)
    return means, cov


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def max_offdiag_symmetric(mat):
    n = len(mat)
    max_val = 0.0
    p = 0
    q = 1
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(mat[i][j])
            if val > max_val:
                max_val = val
                p, q = i, j
    return p, q, max_val


def jacobi_eigendecomposition(mat, tol=1e-12, max_iters=100):
    n = len(mat)
    a = [row[:] for row in mat]
    v = identity_matrix(n)
    for _ in range(max_iters):
        p, q, off = max_offdiag_symmetric(a)
        if off < tol:
            break
        if abs(a[p][p] - a[q][q]) < tol:
            theta = math.pi / 4
        else:
            tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            sign = 1.0 if tau >= 0.0 else -1.0
            t = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
            theta = math.atan(t)
        c = math.cos(theta)
        s = math.sin(theta)

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            if k == p or k == q:
                continue
            aik = a[k][p]
            akq = a[k][q]
            a[k][p] = c * aik - s * akq
            a[p][k] = a[k][p]
            a[k][q] = s * aik + c * akq
            a[q][k] = a[k][q]

        for k in range(n):
            vip = v[k][p]
            viq = v[k][q]
            v[k][p] = c * vip - s * viq
            v[k][q] = s * vip + c * viq

    eigenvalues = [a[i][i] for i in range(n)]
    eigenvectors = [[v[i][j] for i in range(n)] for j in range(n)]
    return eigenvalues, eigenvectors


def spectral_pseudoinverse(mat, eps=1e-12):
    eigenvalues, eigenvectors = jacobi_eigendecomposition(mat)
    n = len(mat)
    inv_vals = [0.0] * n
    positive_vals = []
    for i, val in enumerate(eigenvalues):
        if val > eps:
            inv_vals[i] = 1.0 / val
            positive_vals.append(val)
        else:
            inv_vals[i] = 0.0
    pinv = [[0.0 for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                pinv[i][j] += inv_vals[k] * eigenvectors[k][i] * eigenvectors[k][j]
    cond = float("inf")
    if positive_vals:
        cond = max(positive_vals) / min(positive_vals)
    return pinv, eigenvalues, eigenvectors, cond


def mat_vec(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def mat_mat(a, b):
    rows = len(a)
    cols = len(b[0])
    mid = len(b)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            aik = a[i][k]
            if abs(aik) < 1e-18:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def transpose(mat):
    return [list(row) for row in zip(*mat)]


def solve_linear(mat, rhs):
    size = len(rhs)
    augmented = [row[:] + [rhs[idx]] for idx, row in enumerate(mat)]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(augmented[r][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-12:
            return None
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


def solve_two_asset(i, j, means):
    denom = means[i] - means[j]
    if abs(denom) < 1e-12:
        return None
    w_i = (TARGET_RETURN - means[j]) / denom
    w_j = 1.0 - w_i
    if w_i < -TOL or w_j < -TOL:
        return None
    weights = [0.0] * len(ASSET_NAMES)
    weights[i], weights[j] = w_i, w_j
    return weights


def solve_three_asset_eig(covariance, means):
    n = len(ASSET_NAMES)
    H = [[2.0 * covariance[i][j] for j in range(n)] for i in range(n)]
    H_pinv, eigvals, eigvecs, cond = spectral_pseudoinverse(H)

    A = [
        [1.0 for _ in range(n)],
        [-means[i] for i in range(n)],
    ]
    At = transpose(A)
    M = mat_mat(mat_mat(A, H_pinv), At)
    rhs = [-1.0, TARGET_RETURN]
    y = solve_linear(M, rhs)
    if y is None:
        return None, cond, eigvals, eigvecs

    temp = [sum(At[i][k] * y[k] for k in range(2)) for i in range(n)]
    weights = [-sum(H_pinv[i][j] * temp[j] for j in range(n)) for i in range(n)]
    if any(w < -TOL for w in weights):
        return None, cond, eigvals, eigvecs
    return weights, cond, eigvals, eigvecs


def portfolio_variance(weights, covariance):
    return sum(
        weights[i] * covariance[i][j] * weights[j]
        for i in range(len(ASSET_NAMES))
        for j in range(len(ASSET_NAMES))
    )


def enumerate_candidates(covariance, means):
    candidates = []
    weights, cond, eigvals, eigvecs = solve_three_asset_eig(covariance, means)
    if weights is not None:
        candidates.append(("Eigen decomposition", weights, cond, eigvals, eigvecs))
    for i, j in combinations(range(len(ASSET_NAMES)), 2):
        solution = solve_two_asset(i, j, means)
        if solution is not None:
            candidates.append((f"Two-asset {ASSET_NAMES[i]}-{ASSET_NAMES[j]}", solution, None, None, None))
    return candidates


def best_portfolio(covariance, means):
    candidates = enumerate_candidates(covariance, means)
    feasible = []
    for label, weights, cond, eigvals, eigvecs in candidates:
        expected = sum(means[i] * weights[i] for i in range(len(ASSET_NAMES)))
        if expected + TOL < TARGET_RETURN:
            continue
        if any(w < -TOL for w in weights):
            continue
        variance = portfolio_variance(weights, covariance)
        feasible.append((variance, weights, expected, label, cond, eigvals, eigvecs))
    return min(feasible, key=lambda item: item[0])


def dual_variables(weights, means, covariance):
    sigma_w = [
        sum(covariance[i][j] * weights[j] for j in range(len(ASSET_NAMES)))
        for i in range(len(ASSET_NAMES))
    ]
    active = [idx for idx, w in enumerate(weights) if w > TOL]
    lambda_eq = 0.0
    gamma = 0.0
    if len(active) >= 2:
        i, j = active[0], active[-1]
        numerator = 2.0 * (sigma_w[i] - sigma_w[j])
        denominator = means[i] - means[j]
        gamma = numerator / denominator
        lambda_eq = gamma * means[i] - 2.0 * sigma_w[i]
    slacks = []
    for idx in range(len(ASSET_NAMES)):
        raw = 2.0 * sigma_w[idx] + lambda_eq - gamma * means[idx]
        slacks.append(0.0 if weights[idx] > TOL else max(0.0, raw))
    lagrangian = (
        portfolio_variance(weights, covariance)
        + lambda_eq * (sum(weights) - 1.0)
        + gamma * (TARGET_RETURN - sum(means[i] * weights[i] for i in range(len(ASSET_NAMES))))
        - sum(slacks[i] * weights[i] for i in range(len(ASSET_NAMES)))
    )
    dual_gap = portfolio_variance(weights, covariance) - lagrangian
    return lambda_eq, gamma, slacks, dual_gap


def main():
    returns = compute_returns(PRICE_DATA)
    means, covariance = column_stats(returns)
    (
        variance,
        weights,
        expected,
        label,
        cond,
        eigvals,
        eigvecs,
    ) = best_portfolio(covariance, means)
    allocations = {
        ASSET_NAMES[i]: weights[i] * BUDGET for i in range(len(ASSET_NAMES))
    }
    std_deviation = math.sqrt(variance) * BUDGET
    lambda_eq, gamma, slacks, gap = dual_variables(weights, means, covariance)

    print("Mean returns:", dict(zip(ASSET_NAMES, means)))
    print("Covariance matrix:")
    for row in covariance:
        print(row)
    print("Selected candidate:", label)
    if cond is not None:
        print("Spectral condition number:", cond)
        print("Eigenvalues of 2Q:", eigvals)
    print("Optimal weights:", dict(zip(ASSET_NAMES, weights)))
    print("Dollar allocations:", allocations)
    print("Expected return:", expected)
    print("Portfolio variance:", variance)
    print("Portfolio standard deviation:", std_deviation)
    print("Dual lambda:", lambda_eq)
    print("Dual gamma:", gamma)
    print("Dual slacks:", dict(zip(ASSET_NAMES, slacks)))
    print("Duality gap:", gap)


if __name__ == "__main__":
    main()
