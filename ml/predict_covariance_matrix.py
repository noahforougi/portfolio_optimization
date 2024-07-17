import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import reload

import joblib
import numpy as np
import pandas as pd
from pypfopt import risk_models
from rpy2.robjects import pandas2ri, r
from tqdm import tqdm

import config
import utils
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


pandas2ri.activate()


r_code_dcc_garch = """
library(rmgarch)
library(dplyr)

forecast_dcc_garch_cov <- function(data_window, days_in_month) {
    # Define the GARCH specification
    spec <- ugarchspec(
      variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0))
    )

    # Create the multivariate GARCH specification
    num_columns <- ncol(data_window) - 1
    uspec <- multispec(replicate(num_columns, spec))
    dcc_spec <- dccspec(uspec, dccOrder = c(1, 1), distribution = 'mvnorm')

    # Fit the DCC-GARCH model
    dcc_fit <- dccfit(dcc_spec, data = data_window %>% select(-date))

    if (inherits(dcc_fit, 'uGARCHmultifit')) {
        # Handle non-convergence
        warning('DCC-GARCH fit did not converge. Returning NULL.')
        return(NULL)
    }

    # Forecast the DCC-GARCH model for days_in_month days
    n_ahead <- days_in_month
    dcc_forecast <- dccforecast(dcc_fit, n.ahead = n_ahead)
    dcc_cov_matrix <- rcov(dcc_forecast)[[1]]
    dcc_cov_matrix <- apply(dcc_cov_matrix, c(1, 2), sum)

    return(dcc_cov_matrix)
}
"""

r_code_go_garch = """
library(rmgarch)
library(dplyr)

forecast_go_garch_cov <- function(data_window, days_in_month) {
    spec <- ugarchspec(
        variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0))
    )

    # Create multispec for GO-GARCH
    num_columns <- ncol(data_window) - 1
    uspec <- multispec(replicate(num_columns, spec))
    # Specify the GO-GARCH model
    garch_spec <- gogarchspec(mean.model = 'constant',
                      variance.model = 'goGARCH',
                      distribution.model = 'mvnorm',
                      umodel = uspec)    
    # Fit the GO-GARCH model
    fit <- gogarchfit(spec = garch_spec, data = data_window %>% select(-date))

    if (inherits(fit, 'uGARCHmultifit')) {
        # Handle non-convergence
        warning('GO-GARCH fit did not converge. Returning NULL.')
        return(NULL)
    }

    # Forecast the GO-GARCH model
    n_ahead <- days_in_month
    gogarch_forecast <- gogarchforecast(fit, n.ahead = n_ahead)
    gogarch_cov_matrix <- rcov(gogarch_forecast)[[1]]
    gogarch_cov_matrix <- apply(gogarch_cov_matrix, c(1, 2), sum)

    return(gogarch_cov_matrix)
}
"""
# Execute the R code to define the functions in R environment
r(r_code_dcc_garch)
r(r_code_go_garch)


def forecast_dcc_garch_cov(data_window, days_in_month):
    # Convert the pandas DataFrame to R DataFrame
    r_data_window = pandas2ri.py2rpy(data_window.reset_index())
    dcc_cov_matrix = r["forecast_dcc_garch_cov"](r_data_window, days_in_month)
    if dcc_cov_matrix is None:
        return np.full(
            (data_window.shape[1], data_window.shape[1]), np.nan
        )  # Return NaN matrix if not converged
    return np.array(dcc_cov_matrix)


def forecast_go_garch_cov(data_window, days_in_month):
    # Convert the pandas DataFrame to R DataFrame
    r_data_window = pandas2ri.py2rpy(data_window.reset_index())
    go_garch_cov_matrix = r["forecast_go_garch_cov"](r_data_window, days_in_month)
    if go_garch_cov_matrix is None:
        return np.full(
            (data_window.shape[1], data_window.shape[1]), np.nan
        )  # Return NaN matrix if not converged
    return np.array(go_garch_cov_matrix)


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
    cov_dcc = forecast_dcc_garch_cov(window_df, days_in_next_month)
    cov_gogarch = forecast_go_garch_cov(window_df, days_in_next_month)

    return {
        "date": current_date,
        "true_cov": true_cov,
        "cov_ra": cov_ra,
        "cov_lw_shrinkage": cov_lw_shrinkage,
        "cov_ewma": cov_ewma,
        "cov_dcc": cov_dcc,
        "cov_gogarch": cov_gogarch,
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
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_forecast_cov, df, current_date): current_date
            for current_date in dates
        }
        for future in tqdm(as_completed(futures), total=len(dates)):
            results.append(future.result())

    return results


def main():
    files = utils.list_s3_files(prefix="clean/", bucket_name=config.BUCKET_NAME)
    for f in tqdm(files):
        df = utils.read_s3_file(f)
        results = forecast_cov(df)
        clean_f = f.replace("clean/", "").replace(".csv", "")
        utils.write_s3_joblib(results, f"output/cov_forecasts_{clean_f}.pkl")
