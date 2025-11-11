import math
import random
import time
from pathlib import Path


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


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i]
    return y


def backward_substitution(L, y):
    n = len(L)
    x = [0.0] * n
    for i in reversed(range(n)):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def jacobi_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def jacobi_max_offdiag(mat):
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
    v = jacobi_identity(n)
    for _ in range(max_iters):
        p, q, off = jacobi_max_offdiag(a)
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
    for i, val in enumerate(eigenvalues):
        inv_vals[i] = 1.0 / val if val > eps else 0.0
    pinv = [[0.0 for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                pinv[i][j] += inv_vals[k] * eigenvectors[k][i] * eigenvectors[k][j]
    return pinv


def matrix_inverse(mat):
    """Compute matrix inverse using Gaussian elimination with identity matrix."""
    n = len(mat)
    # Create augmented matrix [A | I]
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
                 for i, row in enumerate(mat)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")

        # Swap rows
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        # Scale pivot row
        pivot = augmented[col][col]
        for j in range(2 * n):
            augmented[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(2 * n):
                augmented[row][j] -= factor * augmented[col][j]

    # Extract inverse from right half of augmented matrix
    inverse = [[augmented[i][j + n] for j in range(n)] for i in range(n)]
    return inverse


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


def mat_vec(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def transpose(mat):
    return [list(row) for row in zip(*mat)]


def solve_gaussian(Q, mu, budget, target):
    n = len(Q)
    size = n + 2
    mat = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = 2.0 * Q[i][j]
    for i in range(n):
        mat[i][n] = 1.0
        mat[i][n + 1] = -mu[i]
        mat[n][i] = 1.0
        mat[n + 1][i] = -mu[i]
    rhs[n] = budget
    rhs[n + 1] = -target
    solution = gaussian_elimination(mat, rhs)
    return solution[:n]


def solve_cholesky(Q, mu, budget, target):
    n = len(Q)
    H = [[2.0 * Q[i][j] for j in range(n)] for i in range(n)]
    L = cholesky_decomposition(H)

    ones = [1.0 for _ in range(n)]
    minus_mu = [-mu[i] for i in range(n)]

    rhs_columns = [ones, minus_mu]
    z_cols = []
    for col in rhs_columns:
        y = forward_substitution(L, col)
        x = backward_substitution(L, y)
        z_cols.append(x)

    M = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            M[i][j] = sum(rhs_columns[i][k] * z_cols[j][k] for k in range(n))

    d = [budget, -target]
    rhs = [-d[0], -d[1]]
    y = gaussian_elimination(
        [[M[0][0], M[0][1]], [M[1][0], M[1][1]]], rhs
    )

    temp = [0.0] * n
    for i in range(n):
        temp[i] = z_cols[0][i] * y[0] + z_cols[1][i] * y[1]

    weights = [-temp[i] for i in range(n)]
    return weights


def solve_eigen(Q, mu, budget, target):
    n = len(Q)
    H = [[2.0 * Q[i][j] for j in range(n)] for i in range(n)]
    H_pinv = spectral_pseudoinverse(H)

    A = [
        [1.0 for _ in range(n)],
        [-mu[i] for i in range(n)],
    ]
    At = transpose(A)
    M = mat_mat(mat_mat(A, H_pinv), At)

    d = [budget, -target]
    rhs = [-d[0], -d[1]]
    y = gaussian_elimination(
        [[M[0][0], M[0][1]], [M[1][0], M[1][1]]], rhs
    )

    temp = [sum(At[i][k] * y[k] for k in range(2)) for i in range(n)]
    weights = [-sum(H_pinv[i][j] * temp[j] for j in range(n)) for i in range(n)]
    return weights


def solve_inverse(Q, mu, budget, target):
    """Solve using direct matrix inverse."""
    n = len(Q)
    H = [[2.0 * Q[i][j] for j in range(n)] for i in range(n)]
    H_inv = matrix_inverse(H)

    A = [
        [1.0 for _ in range(n)],
        [-mu[i] for i in range(n)],
    ]
    At = transpose(A)
    M = mat_mat(mat_mat(A, H_inv), At)

    d = [budget, -target]
    rhs = [-d[0], -d[1]]
    y = gaussian_elimination(
        [[M[0][0], M[0][1]], [M[1][0], M[1][1]]], rhs
    )

    temp = [sum(At[i][k] * y[k] for k in range(2)) for i in range(n)]
    weights = [-sum(H_inv[i][j] * temp[j] for j in range(n)) for i in range(n)]
    return weights


def benchmark():
    sizes = [2,4,8,16,32,64,128]
    budget = 1.0
    target = 0.12
    rows = []

    for idx, n in enumerate(sizes):
        Q = generate_spd_matrix(n, seed=idx)
        mu = [0.2 + 0.02 * (i / n) for i in range(n)]

        start = time.perf_counter()
        weights_gauss = solve_gaussian(Q, mu, budget, target)
        time_gauss = time.perf_counter() - start

        start = time.perf_counter()
        weights_chol = solve_cholesky(Q, mu, budget, target)
        time_chol = time.perf_counter() - start

        start = time.perf_counter()
        weights_eig = solve_eigen(Q, mu, budget, target)
        time_eig = time.perf_counter() - start

        start = time.perf_counter()
        weights_inv = solve_inverse(Q, mu, budget, target)
        time_inv = time.perf_counter() - start

        rows.append(
            {
                "n": n,
                "gaussian_time": time_gauss,
                "cholesky_time": time_chol,
                "eigen_time": time_eig,
                "inverse_time": time_inv,
            }
        )

    return rows


def save_report(rows, svg_path, table_path):
    with open(table_path, "w") as f:
        f.write("n,gaussian_time,cholesky_time,eigen_time,inverse_time\n")
        for row in rows:
            f.write(
                f"{row['n']},{row['gaussian_time']:.6f},{row['cholesky_time']:.6f},{row['eigen_time']:.6f},{row['inverse_time']:.6f}\n"
            )

    width, height = 640, 360
    margin = 60
    xs = [row["n"] for row in rows]
    gauss_vals = [row["gaussian_time"] for row in rows]
    chol_vals = [row["cholesky_time"] for row in rows]
    eig_vals = [row["eigen_time"] for row in rows]
    inv_vals = [row["inverse_time"] for row in rows]

    x_min, x_max = min(xs), max(xs)
    y_max = max(max(gauss_vals), max(chol_vals), max(eig_vals), max(inv_vals)) * 1.1

    def to_svg_coords(x, y):
        scale_x = (width - 2 * margin) / (x_max - x_min)
        scale_y = (height - 2 * margin) / y_max if y_max > 0 else 1.0
        sx = margin + (x - x_min) * scale_x
        sy = height - margin - y * scale_y
        return sx, sy

    def polyline(points, color):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg_lines.append(f'<rect width="100%" height="100%" fill="white"/>')

    svg_lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    svg_lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{margin}" y2="{margin}" stroke="black"/>')

    for k in range(len(xs)):
        x = xs[k]
        sx, _ = to_svg_coords(x, 0.0)
        svg_lines.append(f'<line x1="{sx:.2f}" y1="{height - margin}" x2="{sx:.2f}" y2="{margin}" stroke="#dddddd" stroke-dasharray="4 4"/>')
        svg_lines.append(f'<text x="{sx:.2f}" y="{height - margin + 20}" font-size="12" text-anchor="middle">{x}</text>')

    for t in range(5):
        val = y_max * t / 4
        sx, sy = to_svg_coords(x_min, val)
        svg_lines.append(f'<line x1="{margin}" y1="{sy:.2f}" x2="{width - margin}" y2="{sy:.2f}" stroke="#dddddd" stroke-dasharray="4 4"/>')
        svg_lines.append(f'<text x="{margin - 10}" y="{sy + 4:.2f}" font-size="12" text-anchor="end">{val:.4f}</text>')

    gauss_points = [to_svg_coords(x, y) for x, y in zip(xs, gauss_vals)]
    chol_points = [to_svg_coords(x, y) for x, y in zip(xs, chol_vals)]
    eig_points = [to_svg_coords(x, y) for x, y in zip(xs, eig_vals)]
    inv_points = [to_svg_coords(x, y) for x, y in zip(xs, inv_vals)]
    svg_lines.append(polyline(gauss_points, "#1f77b4"))
    svg_lines.append(polyline(chol_points, "#2ca02c"))
    svg_lines.append(polyline(eig_points, "#d62728"))
    svg_lines.append(polyline(inv_points, "#ff7f0e"))

    for x, y in gauss_points:
        svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1f77b4"/>')
    for x, y in chol_points:
        svg_lines.append(f'<rect x="{x-4:.2f}" y="{y-4:.2f}" width="8" height="8" fill="#2ca02c"/>')
    for x, y in eig_points:
        svg_lines.append(f'<polygon points="{x:.2f},{y-4:.2f} {x+4:.2f},{y+4:.2f} {x-4:.2f},{y+4:.2f}" fill="#d62728"/>')
    for x, y in inv_points:
        svg_lines.append(f'<path d="M {x-4:.2f},{y:.2f} L {x:.2f},{y-4:.2f} L {x+4:.2f},{y:.2f} L {x:.2f},{y+4:.2f} Z" fill="#ff7f0e"/>')

    svg_lines.append('<text x="50%" y="25" font-size="16" text-anchor="middle">Solver Runtime Comparison</text>')
    svg_lines.append('<text x="50%" y="' + str(height - 10) + '" font-size="14" text-anchor="middle">Problem dimension n</text>')
    svg_lines.append('<text transform="translate(' + str(margin - 50) + ' ' + str(height / 2) + ') rotate(-90)" font-size="14" text-anchor="middle">seconds</text>')

    legend_x = margin + 20
    legend_y = margin + 10
    svg_lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="240" height="100" fill="white" stroke="#999999"/>')
    svg_lines.append(f'<line x1="{legend_x + 20}" y1="{legend_y + 20}" x2="{legend_x + 40}" y2="{legend_y + 20}" stroke="#1f77b4" stroke-width="2"/>')
    svg_lines.append(f'<circle cx="{legend_x + 30}" cy="{legend_y + 20}" r="4" fill="#1f77b4"/>')
    svg_lines.append(f'<text x="{legend_x + 50}" y="{legend_y + 24}" font-size="12">Gaussian elimination</text>')
    svg_lines.append(f'<line x1="{legend_x + 20}" y1="{legend_y + 40}" x2="{legend_x + 40}" y2="{legend_y + 40}" stroke="#2ca02c" stroke-width="2"/>')
    svg_lines.append(f'<rect x="{legend_x + 26}" y="{legend_y + 36}" width="8" height="8" fill="#2ca02c"/>')
    svg_lines.append(f'<text x="{legend_x + 50}" y="{legend_y + 44}" font-size="12">Cholesky factor</text>')
    svg_lines.append(f'<line x1="{legend_x + 20}" y1="{legend_y + 60}" x2="{legend_x + 40}" y2="{legend_y + 60}" stroke="#d62728" stroke-width="2"/>')
    svg_lines.append(f'<polygon points="{legend_x + 30},{legend_y + 56} {legend_x + 34},{legend_y + 64} {legend_x + 26},{legend_y + 64}" fill="#d62728"/>')
    svg_lines.append(f'<text x="{legend_x + 50}" y="{legend_y + 64}" font-size="12">Spectral pseudoinverse</text>')
    svg_lines.append(f'<line x1="{legend_x + 20}" y1="{legend_y + 80}" x2="{legend_x + 40}" y2="{legend_y + 80}" stroke="#ff7f0e" stroke-width="2"/>')
    svg_lines.append(f'<path d="M {legend_x + 26},{legend_y + 80} L {legend_x + 30},{legend_y + 76} L {legend_x + 34},{legend_y + 80} L {legend_x + 30},{legend_y + 84} Z" fill="#ff7f0e"/>')
    svg_lines.append(f'<text x="{legend_x + 50}" y="{legend_y + 84}" font-size="12">Direct inverse</text>')

    svg_lines.append('</svg>')

    Path(svg_path).write_text("\n".join(svg_lines))


def main():
    rows = benchmark()
    out_dir = Path(__file__).parent
    save_report(
        rows,
        svg_path=out_dir / "solver_complexity_plot.svg",
        table_path=out_dir / "solver_complexity_table.csv",
    )
    print("n | Gaussian (s) | Cholesky (s) | Eigen (s) | Inverse (s)")
    for row in rows:
        print(
            f"{row['n']:2d} | {row['gaussian_time']:.6f} | {row['cholesky_time']:.6f} | {row['eigen_time']:.6f} | {row['inverse_time']:.6f}"
        )


if __name__ == "__main__":
    main()

