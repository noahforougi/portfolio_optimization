import pandas as pd


def daily_to_monthly(df):
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
    )


def add_lags(df):
    for i in range(1, 13):
        df[f"return_1m_lag{i}"] = df.groupby("ticker")["return_1m"].shift(i)
    return df


def add_label(df):
    df["return_1m_forward"] = df.groupby("ticker")["return_1m"].shift(-1)
    return df


def add_simple_prediction(df):
    df["return_r24m"] = df.groupby("ticker")["return_1m"].transform(
        lambda x: x.rolling(window=24).mean()
    )
    return df


def add_dummy_variables(df, categorical_features):
    df_with_dummies = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_with_dummies


def z_score(df, features):
    z_scored_df = df.copy()
    for feature in features:
        z_scored_df[feature] = z_scored_df.groupby("date")[feature].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    return z_scored_df
