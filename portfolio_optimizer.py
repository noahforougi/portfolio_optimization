import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError

import estimators


class PortfolioOptimizer:
    def __init__(
        self, returns_data, lookback_days, estimator: estimators.Estimator, dates
    ):
        self.data = returns_data
        self.lookback_days = lookback_days
        self.estimator = estimator
        self.dates = dates
        self.previous_weights = None

    def _clean_weights(self, weights):
        d = pd.DataFrame(list(weights.values())).T
        d.columns = list(weights.keys())
        return d

    def perform_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ):
        ef = EfficientFrontier(expected_returns, cov_matrix)
        try:
            weights = ef.max_sharpe(risk_free_rate=0.0)
            return self._clean_weights(weights)
        except:
            if self.previous_weights is not None:
                print("Using previous month's weights due to optimization error")
                return self.previous_weights
            else:
                print("Using equal weights as fallback")
                equal_weights = {
                    asset: 1 / len(expected_returns)
                    for asset in range(len(expected_returns))
                }
                return self._clean_weights(equal_weights)

    def _subset(self, df, current_month) -> pd.DataFrame:
        start = current_month + pd.DateOffset(days=-self.lookback_days)
        end = current_month + pd.DateOffset(days=-1)
        return df.loc[start:end]

    def run(self):
        df = self.data
        results = []
        for d in self.dates:
            print(d)
            df = self._subset(df, d)
            expected_returns = self.estimator.get_expected_returns(df)
            cov_matrix = self.estimator.get_covariance_matrix(df)
            weights = self.perform_optimization(expected_returns, cov_matrix)
            self.previous_weights = (
                weights  # Save current weights to use in next iteration if needed
            )
            results.append(weights)

        return results
