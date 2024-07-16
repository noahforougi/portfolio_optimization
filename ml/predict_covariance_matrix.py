import os
from importlib import reload

import joblib
import pandas as pd
from pypfopt import risk_models
from tqdm import tqdm

import config
from ml import predict_expected_returns, preprocess, train_expected_returns


def forecast_ra_cov(data_window, days_in_month):
    return risk_models.sample_cov(data_window, returns_data=True) / 252 * days_in_month


def forecast_shrinkage_cov(data_window, days_in_month):
    return (
        risk_models.CovarianceShrinkage(data_window, returns_data=True).ledoit_wolf()
        / 252
    ) * days_in_month


def forecast_ewma_cov(data_window, days_in_month, span=180):
    return (
        risk_models.exp_cov(data_window, returns_data=True, span=span) / 252
    ) * days_in_month


def _forecast_cov(df, current_date):
    # Period to predict
    next_date = current_date + pd.offsets.DateOffset(months=1)
    next_month = next_date.month
    next_year = next_date.year

    # True value
    true_df = (
        df.loc[(df.index.month == next_month) & (df.index.year == next_year)].astype(
            float
        )
        / 100
    )
    days_in_next_month = len(true_df)
    true_cov = (
        risk_models.sample_cov(true_df, returns_data=True) / 252 * days_in_next_month
    )

    # This gets the data for the lookback period
    start_date_window = current_date - pd.DateOffset(months=120)
    window_df = (
        df.loc[(df.index >= start_date_window) & (df.index <= current_date)].astype(
            float
        )
        / 100
    )

    # Get the forecast for next month
    cov_ra = forecast_ra_cov(window_df, days_in_next_month)
    cov_lw_shrinkage = forecast_shrinkage_cov(window_df, days_in_next_month)
    cov_ewma = forecast_ewma_cov(window_df, days_in_next_month)

    return {
        "date": current_date,
        "true_cov": true_cov,
        "cov_ra": cov_ra,
        "cov_lw_shrinkage": cov_lw_shrinkage,
        "cov_ewma": cov_ewma,
    }


def forecast_cov(df):
    results = []
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    # Get list of dates to use
    df_filtered = df.loc[config.START_DATE : config.END_DATE]
    dates = (
        df_filtered.groupby([df_filtered.index.year, df_filtered.index.month])
        .apply(lambda x: x.index.max())
        .reset_index(drop=True)
        .tolist()
    )

    # Iterate over the dates
    for current_date in tqdm(dates):
        result = _forecast_cov(df, current_date)
        results.append(result)

    return results


if __name__ == "__main__":
    file_list = os.listdir("../data/clean/")
    for f in tqdm(file_list):
        df = pd.read_csv(f"../data/clean/{f}")
        results = forecast_cov(df)
        clean_f = f.replace(".csv", "")
        joblib.dump(results, f"../data/output/{clean_f}_cov_forecasts.joblib")
