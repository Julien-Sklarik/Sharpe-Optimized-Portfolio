from src.simulation import generate_synthetic, compare_methods

def main():
    R, mu_pop, Sigma_pop, market = generate_synthetic()
    res = compare_methods(R, mu_pop, Sigma_pop, market, rf=0.002)
    print()
    print("Average Sharpe from one synthetic draw")
    for k, v in res.items():
        name = k.replace("_", " ")
        print(f"{name:26s} {v:.4f}")

if __name__ == "__main__":
    main()
