import pandas as pd
from pypfopt import EfficientFrontier
from tqdm import tqdm

import config
import utils


def _optimize(mean_returns, cov_matrix, cov_type, er_type, asset_names):
    try:
        # Find max sharpe ratio portfolio
        ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 0.2))
        ef.max_sharpe(risk_free_rate=0)
        cleaned_weights_max_sharpe = ef.clean_weights()
        weights_series_max_sharpe = pd.Series(
            cleaned_weights_max_sharpe, name="max_sharpe_pyportfolioopt"
        )

        # Find min variance portfolio
        ef = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 0.2))
        ef.min_volatility()
        cleaned_weights_min_variance = ef.clean_weights()
        weights_series_min_variance = pd.Series(
            cleaned_weights_min_variance, name="min_variance_pyportfolioopt"
        )
    except Exception as e:
        print(f"Error encountered: {e}. Using equal weights.")
        equal_weight = 1.0 / len(asset_names)
        weights_series_max_sharpe = pd.Series(
            [equal_weight] * len(asset_names),
            index=asset_names,
            name="max_sharpe_pyportfolioopt",
        )
        weights_series_min_variance = pd.Series(
            [equal_weight] * len(asset_names),
            index=asset_names,
            name="min_variance_pyportfolioopt",
        )

    # Combine the results
    result = pd.DataFrame(
        {
            "asset": asset_names,
            "covariance_type": cov_type,
            "er_type": er_type,
            "max_sharpe_weight": weights_series_max_sharpe,
            "min_variance_weight": weights_series_min_variance,
        }
    )

    return result


def extract_cov_forecast(all_cov_forecasts, date):
    for forecast in all_cov_forecasts:
        if forecast.get("date") == date:
            return forecast
    return None


def _optimize_d(all_cov_forecasts, all_er_forecasts, date, portfolio_order):
    cov_fc = extract_cov_forecast(all_cov_forecasts, date)
    results = list()

    for cov_type in config.COVARIANCE_TYPES:
        for return_type in config.ER_TYPES:
            cov_ra = cov_fc.get(cov_type)

            if cov_ra is None or cov_ra.ndim == 0:
                print(
                    f"Covariance matrix for {cov_type} on {date} is not available. Using equal weights."
                )
                equal_weight = 1.0 / len(portfolio_order)
                result = pd.DataFrame(
                    {
                        "asset": portfolio_order,
                        "covariance_type": cov_type,
                        "er_type": return_type,
                        "max_sharpe_weight": [equal_weight] * len(portfolio_order),
                        "min_variance_weight": [equal_weight] * len(portfolio_order),
                    }
                )
            else:
                if cov_type in ["cov_dcc", "cov_gogarch"]:
                    cov_ra = pd.DataFrame(cov_ra)
                    cov_ra.index = portfolio_order
                    cov_ra.columns = portfolio_order

                mean_returns = (
                    all_er_forecasts.loc[date][["portfolio", return_type]]
                    .reset_index()
                    .pivot(index="date", columns="portfolio", values=return_type)[
                        cov_ra.columns
                    ]
                    .values.flatten()
                )
                cov_matrix = cov_ra.values

                if not (mean_returns > 0).any():
                    print("Expected returns all < 0, using equal weights.")
                    equal_weight = 1.0 / len(portfolio_order)
                    result = pd.DataFrame(
                        {
                            "asset": portfolio_order,
                            "covariance_type": cov_type,
                            "er_type": return_type,
                            "max_sharpe_weight": [equal_weight] * len(portfolio_order),
                            "min_variance_weight": [equal_weight]
                            * len(portfolio_order),
                        }
                    )
                else:
                    result = _optimize(
                        mean_returns, cov_matrix, cov_type, return_type, portfolio_order
                    )

            results.append(result)

    return pd.concat(results).assign(date=date)


def optimize(all_cov_forecasts, all_er_forecasts, portfolio_order):
    dates1 = set([d["date"] for d in all_cov_forecasts])
    dates2 = set(all_er_forecasts.index)
    dates = list(dates1.intersection(dates2))
    dates.sort()

    results = list()
    for d in tqdm(dates):
        result = _optimize_d(all_cov_forecasts, all_er_forecasts, d, portfolio_order)
        results.append(result)
    return pd.concat(results)


def main():
    identifiers = [
        "industry",
        "momentum",
        "size_ltr",
        "size_str",
        "size",
        "sizebtm",
        "sizemomentum",
    ]  # b["btm",

    for identifier in tqdm(identifiers):
        portfolio_order = config.PORTFOLIO_ORDER_DICT.get(identifier)
        # Construct file names
        cov_forecast_file = f"output/cov_forecasts_{identifier}.pkl"
        er_forecast_file = f"output/er_forecasts_{identifier}.csv"
        output_file = f"output/optimal_weights_{identifier}.csv"

        # Read covariance forecasts
        cov_forecasts = utils.read_s3_joblib(cov_forecast_file)

        # Read expected returns forecasts
        er_forecasts = utils.read_s3_file(er_forecast_file)
        er_forecasts["date"] = pd.to_datetime(er_forecasts["date"])
        er_forecasts = er_forecasts.set_index("date")

        # Perform optimization
        res = optimize(cov_forecasts, er_forecasts, portfolio_order)

        # Write results to S3
        utils.write_s3_file(res, output_file)
