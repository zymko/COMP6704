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


def cholesky_decomposition(mat):
    n = len(mat)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = mat[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def cholesky_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i]

    x = [0.0] * n
    for i in reversed(range(n)):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def cholesky_solve_multiple(L, rhs_columns):
    return [cholesky_solve(L, column) for column in rhs_columns]


def matrix_vector(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


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


def spectral_condition_number(H, L):
    n = len(H)

    def rayleigh(vec):
        Hv = matrix_vector(H, vec)
        num = sum(vec[i] * Hv[i] for i in range(n))
        den = sum(vec[i] * vec[i] for i in range(n))
        return num / den

    def normalize(vec):
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    v = normalize([1.0 for _ in range(n)])
    for _ in range(40):
        v = normalize(matrix_vector(H, v))
    lambda_max = rayleigh(v)

    v = normalize([1.0 for _ in range(n)])
    for _ in range(40):
        v = cholesky_solve(L, v)
        v = normalize(v)
    lambda_min = rayleigh(v)

    if lambda_min <= 0.0:
        return float("inf")
    return lambda_max / lambda_min


def solve_three_asset_cholesky(covariance, means):
    n = len(ASSET_NAMES)
    H = [[2.0 * covariance[i][j] for j in range(n)] for i in range(n)]
    L = cholesky_decomposition(H)

    A = [
        [1.0 for _ in range(n)],
        [-means[i] for i in range(n)],
    ]
    b = [1.0, -TARGET_RETURN]

    rhs_cols = [[A[row][i] for i in range(n)] for row in range(len(A))]
    z_cols = cholesky_solve_multiple(L, rhs_cols)

    M = [[0.0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A)):
            M[i][j] = sum(A[i][k] * z_cols[j][k] for k in range(n))

    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if abs(det) < 1e-12:
        return None, None

    rhs = [-2.0 * b[0], -2.0 * b[1]]
    y0 = (rhs[0] * M[1][1] - M[0][1] * rhs[1]) / det
    y1 = (M[0][0] * rhs[1] - rhs[0] * M[1][0]) / det

    weights = [
        -0.5 * (z_cols[0][i] * y0 + z_cols[1][i] * y1) for i in range(n)
    ]
    if any(w < -TOL for w in weights):
        return None, None

    cond = spectral_condition_number(H, L)
    return weights, cond


def portfolio_variance(weights, covariance):
    return sum(
        weights[i] * covariance[i][j] * weights[j]
        for i in range(len(ASSET_NAMES))
        for j in range(len(ASSET_NAMES))
    )


def enumerate_candidates(covariance, means):
    candidates = []
    three_asset, cond = solve_three_asset_cholesky(covariance, means)
    if three_asset is not None:
        candidates.append(("Cholesky", three_asset, cond))
    for i, j in combinations(range(len(ASSET_NAMES)), 2):
        solution = solve_two_asset(i, j, means)
        if solution is not None:
            candidates.append((f"Two-asset {ASSET_NAMES[i]}-{ASSET_NAMES[j]}", solution, None))
    return candidates


def best_portfolio(covariance, means):
    candidates = enumerate_candidates(covariance, means)
    feasible = []
    for label, weights, cond in candidates:
        expected = sum(means[i] * weights[i] for i in range(len(ASSET_NAMES)))
        if expected + TOL < TARGET_RETURN:
            continue
        if any(w < -TOL for w in weights):
            continue
        variance = portfolio_variance(weights, covariance)
        feasible.append((variance, weights, expected, label, cond))
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
    variance, weights, expected, label, cond = best_portfolio(covariance, means)
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
        print("Condition number (2-norm estimate):", cond)
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

