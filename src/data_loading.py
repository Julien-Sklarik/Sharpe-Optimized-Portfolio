from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_market_excess(name="MarketExcessReturns.csv"):
    p = DATA_DIR / name
    x = pd.read_csv(p, header=None).values.flatten()
    return x

def load_asset_excess(name="MonthlyExcessReturns.csv"):
    p = DATA_DIR / name
    x = pd.read_csv(p, header=None)
    return x

def load_truth_mu(name="ExpectedExcessReturns.csv"):
    p = DATA_DIR / name
    x = pd.read_csv(p, header=None).values.flatten()
    return x

def load_truth_sigma(name="CovarianceMatrix.csv"):
    p = DATA_DIR / name
    x = pd.read_csv(p, header=None).values
    return x

def validate_dimensions(returns_df):
    T, N = returns_df.shape
    if T < 12 or N < 2:
        raise ValueError("need at least one year of monthly data and at least two assets")
    return T, N
