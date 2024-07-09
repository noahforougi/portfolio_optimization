from typing import Optional

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import exp_cov, sample_cov
from sklearn.ensemble import RandomForestRegressor


class Estimator:
    def get_expected_returns(self, df):
        raise NotImplementedError

    def get_covariance_matrix(self, df):
        raise NotImplementedError

    def get_ewma_covariance_matrix(self, df):
        raise NotImplementedError


class HistoricalMeanEstimator(Estimator):
    def get_expected_returns(self, df):
        return mean_historical_return(df, returns_data=True)

    def get_covariance_matrix(self, df):
        return sample_cov(df, returns_data=True)

    def get_ewma_covariance_matrix(self, df):
        return exp_cov(df, span=180, returns_data=True)
