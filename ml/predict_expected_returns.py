import pandas as pd

from ml import train_expected_returns


def expected_return_forecasts(df, model_dict, dates):
    monthly_predictions = {}

    for date in pd.to_datetime(dates):
        year = date.year
        month = date.month

        if year in model_dict:
            model_for_year = model_dict[year]
        else:
            continue

        df_month = df[(df.index.year == year) & (df.index.month == month)].sort_values(
            "portfolio"
        )
        if not df_month.empty:
            features = train_expected_returns.get_features(df_month)
            X_test = df_month[features]
            monthly_predictions_for_period = {}
            for model_type, model in model_for_year.items():
                if model_type == train_expected_returns.ModelType.ROLLING_AVERAGE:
                    lookback_data = train_expected_returns.get_lookback_period_monthly(
                        df, date
                    )
                    rolling_avg = lookback_data.groupby("portfolio")["return_1m"].mean()
                    monthly_predictions_for_period[model_type] = df_month[
                        "portfolio"
                    ].map(rolling_avg)
                elif model_type == train_expected_returns.ModelType.EWMA:
                    lookback_data = train_expected_returns.get_lookback_period_monthly(
                        df, date
                    )
                    ewma = lookback_data.groupby("portfolio")["return_1m"].apply(
                        lambda x: x.ewm(span=10).mean().iloc[-1]
                    )
                    monthly_predictions_for_period[model_type] = df_month[
                        "portfolio"
                    ].map(ewma)
                elif model_type == train_expected_returns.ModelType.ARIMA:
                    pass
                elif model is None:
                    pass
                else:
                    monthly_predictions_for_period[model_type] = model.predict(X_test)
            monthly_predictions_df = pd.DataFrame(
                monthly_predictions_for_period, index=df_month.index
            )
            monthly_predictions_df["portfolio"] = df_month["portfolio"]
            monthly_predictions_df["true_value"] = df_month["return_1m_forward"]
            monthly_predictions_df["date"] = df_month.index
            monthly_predictions[date] = monthly_predictions_df

    all_predictions_df = pd.concat(
        monthly_predictions.values(), keys=monthly_predictions.keys()
    ).reset_index(drop=True)
    cols = ["date", "portfolio", "true_value"] + [
        col
        for col in all_predictions_df.columns
        if col not in ["date", "portfolio", "true_value"]
    ]
    all_predictions_df = all_predictions_df[cols].sort_values(["date", "portfolio"])

    return all_predictions_df
