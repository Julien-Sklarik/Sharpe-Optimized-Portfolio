import numpy as np
from src.metrics import portfolio_sigma, sharpe_from_params

def test_sigma_is_scalar():
    w = np.array([0.6, 0.4])
    S = np.array([[0.04, 0.01],
                  [0.01, 0.09]])
    s = portfolio_sigma(w, S)
    assert isinstance(s, float)

def test_sharpe_increases_with_mean():
    w = np.array([0.5, 0.5])
    S = np.eye(2) * 0.04
    rf = 0.001
    s1 = sharpe_from_params(w, np.array([0.005, 0.005]), S, rf)
    s2 = sharpe_from_params(w, np.array([0.006, 0.006]), S, rf)
    assert s2 > s1
