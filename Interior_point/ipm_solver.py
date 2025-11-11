import numpy as np
import pandas as pd
import cvxpy as cp

# --- Step 1: Parse and Prepare Data ---

# Raw stock price data (November 2000 - November 2001)
price_data = [
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
    [115.590, 55.081, 1.800]
]

# Load data into a Pandas DataFrame for easier calculation
df = pd.DataFrame(price_data, columns=['IBM', 'WMT', 'SEHI'])

# Calculate monthly returns: (p_t - p_{t-1}) / p_{t-1}
monthly_returns = df.pct_change().dropna()

# --- Step 2: Calculate Model Parameters (r̄ and Q) ---

# r_bar: Expected monthly return vector (mean of historical data)
r_bar = monthly_returns.mean().values

# Q: Covariance matrix of monthly returns
Q = monthly_returns.cov().values

print("--- Data Calculation Results ---")
print("Stock Tickers:", df.columns.tolist())
print(f"\nExpected Monthly Returns (r̄):\n{r_bar}")
print(f"\nReturn Covariance Matrix (Q):\n{Q}")
print("-" * 30 + "\n")


# --- Step 3: Build and Solve the Model using CVXPY and Interior Point Method ---

# 1. Define Optimization Variable
# x is a vector with 3 elements, representing the investment amount in each stock
n = len(df.columns)
x = cp.Variable(n)

# 2. Define Objective Function and Constraints
# Objective: Minimize portfolio variance (risk) x'Qx
objective = cp.Minimize(cp.quad_form(x, Q))

# Constraint list
constraints = [
    cp.sum(x) <= 1000,   # Total investment amount must not exceed $1000
    r_bar @ x >= 50,     # Expected monthly return must be at least $50 (5% of $1000)
    x >= 0               # No short selling allowed (investment amounts are non-negative)
]

# 3. Create and Solve the Problem
# CVXPY automatically calls an efficient solver (e.g., ECOS),
# which uses the Interior Point Method to solve this quadratic programming problem.
problem = cp.Problem(objective, constraints)
problem.solve()

# --- Step 4: Interpret and Display Results ---

print("--- Optimization Results ---")
print(f"Solver Status: {problem.status}")

# Check if the solution was successful
if problem.status == cp.OPTIMAL:
    # Extract the optimal solution
    optimal_x = x.value

    # Calculate portfolio expected return and risk
    expected_return = r_bar @ optimal_x
    portfolio_variance = optimal_x.T @ Q @ optimal_x
    portfolio_std_dev = np.sqrt(portfolio_variance)

    print("\n[Optimal Investment Strategy]")
    print("To minimize risk while achieving an expected return of at least $50, the suggested capital allocation is:")
    for i, ticker in enumerate(df.columns):
        print(f"  - {ticker}: ${optimal_x[i]:.2f}")

    total_investment = np.sum(optimal_x)
    print(f"\nTotal Investment: ${total_investment:.2f}")

    print("\n[Expected Portfolio Performance]")
    print(f"  - Expected Monthly Return: ${expected_return:.2f}")
    print(f"  - Minimized Portfolio Variance (Risk): {portfolio_variance:.4f}")
    print(f"  - Minimized Portfolio Standard Deviation (Risk): {portfolio_std_dev:.4f}")

    # Calculate and display investment weights
    if total_investment > 1e-6:
        weights = optimal_x / total_investment
        print("\n[Investment Portfolio Weights]")
        for i, ticker in enumerate(df.columns):
            print(f"  - {ticker}: {weights[i]*100:.2f}%")

else:
    print("\nFailed to find an optimal solution. Possible reason:")
    print("Based on the historical data provided, it may not be possible for any portfolio to achieve the 5% expected return under finite risk.")

print("-" * 25)