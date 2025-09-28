import numpy as np
from .portfolio import (
    equal_weight,
    mean_variance_weights,
    min_variance_weights,
    jsj_weights,
    capm_weights,
    best_constrained_sharpe,
)
from .metrics import sharpe_from_series, sharpe_from_params

def generate_synthetic(N=50, T=240, rf=0.002, re_M=0.008, q=0.05, seed=123):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, N)
    alpha = rng.uniform(0.1, 0.3, N)
    factor = rng.normal(0.0, q, T)
    R = np.zeros((T, N))
    for i in range(N):
        eps = rng.normal(0.0, 1.0, T)
        R[:, i] = rf + re_M * beta[i] + beta[i] * factor + alpha[i] * eps
    mu_pop = rf + re_M * beta
    Sigma_pop = np.diag(alpha ** 2) + (q ** 2) * np.outer(beta, beta)
    market_excess = factor + re_M
    return R, mu_pop, Sigma_pop, market_excess

def sample_moments(R):
    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False)
    return mu, S

def compare_methods(R, mu_pop, Sigma_pop, market_excess, rf=0.002):
    T, N = R.shape
    mu_s, S_s = sample_moments(R)

    S_market = sharpe_from_series(market_excess, rf)

    w_eq = equal_weight(N)
    S_eq = sharpe_from_params(w_eq, mu_pop, Sigma_pop, rf)

    w_mv = mean_variance_weights(mu_s, S_s)
    S_mv = sharpe_from_params(w_mv, mu_pop, Sigma_pop, rf)

    w_min = min_variance_weights(S_s)
    S_min = sharpe_from_params(w_min, mu_pop, Sigma_pop, rf)

    w_jsj = jsj_weights(mu_s, S_s, T)
    S_jsj = sharpe_from_params(w_jsj, mu_pop, Sigma_pop, rf)

    w_capm = capm_weights(
        returns_df=_to_df(R),
        market_excess=market_excess,
        sigma_sample=S_s,
        risk_free=rf,
    )
    S_capm = sharpe_from_params(w_capm, mu_pop, Sigma_pop, rf)

    w_cmv, S_cmv = best_constrained_sharpe(mu_s, S_s, mu_pop, Sigma_pop, rf)

    return {
        "market": S_market,
        "one_over_N": S_eq,
        "mean_variance": S_mv,
        "min_variance": S_min,
        "jsj": S_jsj,
        "capm": S_capm,
        "cmv_no_short": S_cmv,
    }

def _to_df(R):
    import pandas as pd
    return pd.DataFrame(R)
