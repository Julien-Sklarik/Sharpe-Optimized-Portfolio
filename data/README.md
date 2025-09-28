Expected files if you want to run on real data

MarketExcessReturns.csv          shape T by 1                      units decimal monthly excess return
MonthlyExcessReturns.csv         shape T by N                      units decimal monthly excess returns for N assets
ExpectedExcessReturns.csv        shape N                           units decimal population means for sanity checks
CovarianceMatrix.csv             shape N by N                      units monthly covariance for sanity checks

The code treats these as reference truth only for evaluation and uses sample estimates for portfolio choice
Place files in this folder exactly with the names above
