import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from scipy.optimize import minimize
import operator as op

def _safe_inv(mat):
    try:
        return inv(mat)
    except LinAlgError:
        return pinv(mat)

def equal_weight(n_assets):
    w = np.ones(n_assets) / n_assets
    return w

def mean_variance_weights(mu_sample, sigma_sample):
    invS = _safe_inv(sigma_sample)
    one = np.ones(mu_sample.shape[0])
    num = invS @ mu_sample
    den = one @ num
    return num / den

def min_variance_weights(sigma_sample):
    invS = _safe_inv(sigma_sample)
    one = np.ones(sigma_sample.shape[0])
    num = invS @ one
    den = one @ num
    return num / den

def jsj_weights(mu_sample, sigma_sample, T):
    one = np.ones(mu_sample.shape[0])
    invS = _safe_inv(sigma_sample)
    num_mv = invS @ mu_sample
    num_min = invS @ one
    mu_0 = (one @ num_mv) / (one @ num_min)
    diff = mu_sample - mu_0
    v_num = mu_sample.shape[0] + 2.0
    v_den = mu_sample.shape[0] + 2.0 + T * (diff @ invS @ diff)
    v = v_num / v_den
    mu_shrunk = (1.0 - v) * mu_sample + v * mu_0
    num = invS @ mu_shrunk
    den = one @ num
    return num / den

def capm_weights(returns_df, market_excess, sigma_sample, risk_free):
    T, N = returns_df.shape
    R = returns_df.values
    M = market_excess
    M_center = M - M.mean()
    denom = float(np.var(M_center))
    if denom == 0.0:
        betas = np.zeros(N)
    else:
        R_center = R - R.mean(axis=0, keepdims=True)
        covs = (R_center * M_center.reshape(-1, 1)).mean(axis=0)
        betas = covs / denom
    mu_capm = risk_free + betas * (M.mean() - risk_free)
    invS = _safe_inv(sigma_sample)
    one = np.ones(N)
    num = invS @ mu_capm
    den = one @ num
    return num / den

def _variance(w, sigma):
    return float(w @ sigma @ w)

def _target_constraints(mu_est, target_mu):
    return [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "eq", "fun": lambda w: float(op.sub(float(w @ mu_est), float(target_mu)))},
    ]

def min_var_no_short_for_target(mu_est, sigma_est, target_mu):
    N = mu_est.shape[0]
    bounds = [(0.0, None)] * N
    w0 = np.ones(N) / N
    res = minimize(lambda w: _variance(w, sigma_est),
                   w0,
                   method="SLSQP",
                   bounds=bounds,
                   constraints=_target_constraints(mu_est, target_mu),
                   options={"maxiter": 200, "ftol": 1e-6})
    if not res.success:
        return None
    return res.x

def best_constrained_sharpe(mu_sample, sigma_sample, mu_true, sigma_true, risk_free, grid_size=50):
    one = np.ones(mu_sample.shape[0])
    lo = float(np.min(mu_sample))
    hi = float(np.max(mu_sample))
    best_S = 0.0
    best_w = one / one.shape[0]
    for target in np.linspace(lo, hi, grid_size):
        w = min_var_no_short_for_target(mu_sample, sigma_sample, target)
        if w is None:
            continue
        m = float(w @ mu_true)
        s = float(np.sqrt(w @ sigma_true @ w))
        S = op.sub(m, risk_free) / s
        if S > best_S:
            best_S = S
            best_w = w
    return best_w, best_S
