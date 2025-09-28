import numpy as np
import operator as op

def sharpe_from_series(excess_series, risk_free):
    m = float(np.mean(excess_series))
    s = float(np.std(excess_series, ddof=0))
    num = op.sub(m, risk_free)
    if s == 0.0:
        return 0.0
    return num / s

def portfolio_sigma(weights, sigma):
    return float(np.sqrt(weights @ sigma @ weights))

def sharpe_from_params(weights, mu, sigma, risk_free):
    m = float(weights @ mu)
    s = portfolio_sigma(weights, sigma)
    return op.sub(m, risk_free) / s
