import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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
    # Remove NaN rows
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Linear regression of returns on macro
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
    return T_stat
