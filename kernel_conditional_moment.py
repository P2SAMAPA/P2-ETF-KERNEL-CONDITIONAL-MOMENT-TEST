import numpy as np
from sklearn.preprocessing import StandardScaler

def rbf_kernel(X, Y, sigma=1.0):
    """Gaussian kernel between two matrices."""
    # For simplicity, we assume X and Y are 1D or 2D? We'll compute pairwise.
    # For a single test, we need a scalar statistic. We'll compute:
    # T = (1/n^2) sum_{i,j} K(X_i, X_j) * epsilon_i * epsilon_j, where epsilon = residual from linear regression.
    pass

def kernel_conditional_moment_test(returns, macro_df, sigma=1.0, num_permutations=100):
    """
    Test whether macro variables have predictive power beyond linear models.
    Steps:
      1. Regress returns on macro (linear model) to get residuals.
      2. Compute test statistic T = (1/n^2) * sum_{i,j} K(macro_i, macro_j) * resid_i * resid_j.
      3. Higher T indicates non‑linear predictability.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Linear regression of returns on macro
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(macro_scaled, returns)
    resid = returns - lr.predict(macro_scaled)
    # Compute test statistic T
    n = len(macro_scaled)
    T_stat = 0.0
    # Gaussian kernel on macro
    for i in range(n):
        for j in range(n):
            diff = macro_scaled[i] - macro_scaled[j]
            K_ij = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
            T_stat += K_ij * resid[i] * resid[j]
    T_stat = T_stat / (n**2)
    # T_stat can be negative due to finite sample; take absolute value
    T_stat = abs(T_stat)
    # Optional: permutation test for p‑value (not used as score)
    # For score, we just use T_stat
    return T_stat
