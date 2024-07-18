from typing import List

import pandas as pd


def daily_to_monthly(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.melt(ignore_index=False, var_name="ticker", value_name="return_1d")
    df["month"] = df.index.to_period("M")
    df["year"] = df.index.year
    return (
        df.reset_index()
        .sort_values(["date", "ticker"])
        .groupby(["ticker", "month"])
        .agg(
            return_1m=("return_1d", lambda x: (1 + x / 100).prod() - 1),
            avg_daily_return=("return_1d", "mean"),
            stddev_return=("return_1d", "std"),
            date=("date", "max"),
        )
        .reset_index()
        .drop(columns=["month"])
        .set_index("date")
        .sort_index()
        .sort_values("date")
        .assign(return_1m_forward=lambda x: x.groupby("ticker").return_1m.shift(-1))
    )


def add_lags(df: pd.DataFrame):
    for i in range(1, 13):
        df[f"return_1m_lag{i}"] = df.groupby("ticker")["return_1m"].shift(i)
    return df


def add_label(df: pd.DataFrame):
    df["return_1m_forward"] = df.groupby("ticker")["return_1m"].shift(-1)
    return df


def add_simple_prediction(df: pd.DataFrame):
    df["return_r24m"] = df.groupby("ticker")["return_1m"].transform(
        lambda x: x.rolling(window=24).mean()
    )
    return df


def calculate_momentum(df: pd.DataFrame, periods: List[int] = [3, 6, 12]):
    df = df.copy()
    for period in periods:
        df = df.reset_index()
        momentum_feature = (
            df.groupby("ticker")["return_1m"]
            .rolling(window=period)
            .apply(lambda x: (x + 1).prod() - 1)
        )
        df = df.set_index(["date", "ticker"])
        df[f"momentum_{period}m"] = momentum_feature.values

    return df


def add_dummy_variables(df, categorical_features):
    df["portfolio"] = df["ticker"]
    df_with_dummies = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_with_dummies


def z_score(df, features):
    z_scored_df = df.copy()
    for feature in features:
        z_scored_df[feature] = z_scored_df.groupby("date")[feature].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    return z_scored_df


def prepare_training_data(df: pd.DataFrame):
    df = daily_to_monthly(df)
    df = add_lags(df)
    df = add_simple_prediction(df)
    df = calculate_momentum(df).reset_index().set_index("date")
    df = add_dummy_variables(df, ["ticker"])
    return df.sort_index()


def prepare_training_data_no_dummys(df: pd.DataFrame):
    df = daily_to_monthly(df)
    df = add_lags(df)
    df = add_simple_prediction(df)
    df = calculate_momentum(df).reset_index().set_index("date")
    return df.sort_index()
