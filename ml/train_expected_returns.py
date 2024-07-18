import logging
from enum import Enum

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

import config
import utils
from ml import preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_preprocessing_pipeline(degree=2):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("poly", PolynomialFeatures(degree=degree)),
            ("scaler", StandardScaler()),
        ]
    )


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
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
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


def train_rolling_average_model(y_train):
    return None


def train_ewma_model(y_train):
    return None


def train_arima_model(y_train):
    return None


def train_random_forest_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline(
        [
            ("preprocess", preprocessing_pipeline),
            ("rf", RandomForestRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [10, 20],
        "rf__min_samples_split": [2, 10],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_xgboost_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline(
        [("preprocess", preprocessing_pipeline), ("xgb", XGBRegressor(random_state=42))]
    )

    param_grid = {
        "xgb__n_estimators": [100, 300],
        "xgb__max_depth": [3, 9],
        "xgb__learning_rate": [0.01, 0.1],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_linear_regression_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline(
        [("preprocess", preprocessing_pipeline), ("linear", LinearRegression())]
    )

    model.fit(X_train, y_train)
    return model


def train_ridge_regression_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline([("preprocess", preprocessing_pipeline), ("ridge", Ridge())])

    param_grid = {"ridge__alpha": [0.1, 1, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_lasso_regression_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline([("preprocess", preprocessing_pipeline), ("lasso", Lasso())])

    param_grid = {"lasso__alpha": [0.01, 0.1, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_svr_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline([("preprocess", preprocessing_pipeline), ("svr", SVR())])

    param_grid = {
        "svr__C": [0.1, 1, 10],
        "svr__gamma": [0.001, 0.01],
        "svr__kernel": ["linear", "rbf"],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_gradient_boosting_model(X_train, y_train):
    preprocessing_pipeline = create_preprocessing_pipeline()
    model = Pipeline(
        [
            ("preprocess", preprocessing_pipeline),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "gbr__n_estimators": [100, 200],
        "gbr__learning_rate": [0.01, 0.1],
        "gbr__max_depth": [3, 7],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_expected_returns(df):
    label = get_label()
    features = get_features(df)
    models_dict = {}

    for year in tqdm(range(config.START_YEAR, config.END_YEAR)):
        train_data = get_lookback_period(df, year)
        X_train = train_data[features]
        y_train = train_data[label]

        logger.info("Training Model --- Rolling Average")
        rolling_avg_model = train_rolling_average_model(y_train)
        logger.info("Training Model --- EWMA")
        ewma_model = train_ewma_model(y_train)
        logger.info("Training Model --- ARIMA")
        arima_model = train_arima_model(y_train)
        logger.info("Training Model --- Random Forest")
        rf_model = train_random_forest_model(X_train, y_train)
        logger.info("Training Model --- XGBoost")
        xgboost_model = train_xgboost_model(X_train, y_train)
        logger.info("Training Model --- Linear Regression")
        lin_reg_model = train_linear_regression_model(X_train, y_train)
        logger.info("Training Model --- Ridge Regression")
        ridge_regression_model = train_ridge_regression_model(X_train, y_train)
        logger.info("Training Model --- Lasso Regression")
        lasso_regression_model = train_lasso_regression_model(X_train, y_train)
        logger.info("Training Model --- SVR Model")
        svr_model = train_svr_model(X_train, y_train)
        logger.info("Training Model --- Gradient Boosting Model")
        gradient_boosting_model = train_gradient_boosting_model(X_train, y_train)
        models_dict[year] = {
            ModelType.ROLLING_AVERAGE: rolling_avg_model,
            ModelType.EWMA: ewma_model,
            ModelType.ARIMA: arima_model,
            ModelType.RANDOM_FOREST: rf_model,
            ModelType.XGBOOST: xgboost_model,
            ModelType.LINEAR_REGRESSION: lin_reg_model,
            ModelType.RIDGE_REGRESSION: ridge_regression_model,
            ModelType.LASSO_REGRESSION: lasso_regression_model,
            ModelType.SVR: svr_model,
            ModelType.GRADIENT_BOOSTING: gradient_boosting_model,
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
