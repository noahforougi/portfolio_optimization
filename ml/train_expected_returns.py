from enum import Enum

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

import config
import utils
from ml import preprocess


def get_label():
    return "return_1m_forward"


def get_features(df):
    base_features = [
        "avg_daily_return",
        "stddev_return",
        "return_1m_lag1",
        "return_1m_lag2",
        "return_1m_lag3",
        "return_1m_lag4",
        "return_1m_lag5",
        "return_1m_lag6",
        "return_1m_lag7",
        "return_1m_lag8",
        "return_1m_lag9",
        "return_1m_lag10",
        "return_1m_lag11",
        "return_1m_lag12",
    ]
    ticker_features = [col for col in df.columns if "ticker" in col]
    return base_features + ticker_features


def get_lookback_period(df, year, lookback_years=10):
    start_year = year - lookback_years
    end_year = year - 1
    return df[(df.index.year >= start_year) & (df.index.year <= end_year)]


def get_lookback_period_monthly(df, date, lookback_years=10):
    start = date + pd.DateOffset(years=-lookback_years)
    return df[(df.index >= start) & (df.index <= date)]


class ModelType(Enum):
    ROLLING_AVERAGE = "rolling_average"
    EWMA = "ewma"
    ARIMA = "arima"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    SVR = "svr"
    GRADIENT_BOOSTING = "gradient_boosting"


def train_rolling_average_model(y_train):
    return None


def train_ewma_model(y_train):
    return None


def train_arima_model(y_train):
    return None


def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression_model(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model


def train_lasso_regression_model(X_train, y_train):
    model = Lasso()
    model.fit(X_train, y_train)
    return model


def train_svr_model(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_model(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model


def train_expected_returns(df):
    label = get_label()
    features = get_features(df)
    models_dict = {}

    for year in range(config.START_YEAR, config.END_YEAR):
        train_data = get_lookback_period(df, year)
        X_train = train_data[features]
        y_train = train_data[label]

        models_dict[year] = {
            ModelType.ROLLING_AVERAGE: train_rolling_average_model(y_train),
            ModelType.EWMA: train_ewma_model(y_train),
            ModelType.ARIMA: train_arima_model(y_train),
            ModelType.RANDOM_FOREST: train_random_forest_model(X_train, y_train),
            ModelType.XGBOOST: train_xgboost_model(X_train, y_train),
            ModelType.LINEAR_REGRESSION: train_linear_regression_model(
                X_train, y_train
            ),
            ModelType.RIDGE_REGRESSION: train_ridge_regression_model(X_train, y_train),
            ModelType.LASSO_REGRESSION: train_lasso_regression_model(X_train, y_train),
            ModelType.SVR: train_svr_model(X_train, y_train),
            ModelType.GRADIENT_BOOSTING: train_gradient_boosting_model(
                X_train, y_train
            ),
        }
    return models_dict


def main():
    files = utils.list_s3_files(prefix="clean/", bucket_name=config.BUCKET_NAME)
    for f in tqdm(files):
        df = utils.read_s3_file(f)
        df = preprocess.prepare_training_data(df)
        model_dict = train_expected_returns(df)
        clean_file = f.replace(".csv", "").replace("clean/", "")
        utils.write_s3_joblib(model_dict, f"return_models/{clean_file}_er_models.pkl")


if __name__ == "__main__":
    main()
