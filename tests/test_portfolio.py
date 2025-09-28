import numpy as np
from src.portfolio import (
    equal_weight,
    mean_variance_weights,
    min_variance_weights,
    jsj_weights,
)

def test_equal_weight_sums_to_one():
    w = equal_weight(10)
    assert np.isclose(w.sum(), 1.0)

def test_min_variance_sums_to_one():
    S = np.eye(3) * 0.04
    w = min_variance_weights(S)
    assert np.isclose(w.sum(), 1.0)

def test_mean_variance_sums_to_one():
    mu = np.array([0.01, 0.02, 0.03])
    S = np.eye(3) * 0.05
    w = mean_variance_weights(mu, S)
    assert np.isclose(w.sum(), 1.0)

def test_jsj_sums_to_one():
    mu = np.array([0.01, 0.02, 0.03])
    S = np.eye(3) * 0.05
    w = jsj_weights(mu, S, T=120)
    assert np.isclose(w.sum(), 1.0)
