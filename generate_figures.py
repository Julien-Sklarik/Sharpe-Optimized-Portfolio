import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# allow running from repo root
root = Path(__file__).resolve().parents[0]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.simulation import generate_synthetic, compare_methods, sample_moments
from src.portfolio import equal_weight, best_constrained_sharpe

def make_figures(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    R, mu_pop, Sigma_pop, market = generate_synthetic()
    res = compare_methods(R, mu_pop, Sigma_pop, market, rf=0.002)

    # bar of sharpes
    names = list(res.keys())
    vals = [res[k] for k in names]

    plt.figure()
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=30, ha="right")
    plt.ylabel("Sharpe")
    plt.title("Sharpe by method")
    plt.tight_layout()
    plt.savefig(out_dir / "sharpe_by_method.png", dpi=200)
    plt.close()

    # cumulative returns for equal weight vs CMV no short
    mu_s, S_s = sample_moments(R)
    w_eq = equal_weight(R.shape[1])
    w_cmv, _ = best_constrained_sharpe(mu_s, S_s, mu_pop, Sigma_pop, risk_free=0.002)

    rets_eq = R @ w_eq
    rets_cmv = R @ w_cmv

    def to_cum(ret_series):
        wealth = np.cumprod(1.0 + ret_series)
        return wealth

    cum_eq = to_cum(rets_eq)
    cum_cmv = to_cum(rets_cmv)

    plt.figure()
    plt.plot(cum_eq, label="equal weight")
    plt.plot(cum_cmv, label="cmv no short")
    plt.xlabel("time")
    plt.ylabel("cumulative wealth")
    plt.title("Cumulative wealth comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_returns.png", dpi=200)
    plt.close()

    return res

if __name__ == "__main__":
    out = root / "assets"
    make_figures(out)
