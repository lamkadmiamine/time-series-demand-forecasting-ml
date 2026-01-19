import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from models.base_metrics import ForecastMetrics


class BenchmarkMLModels:
    """
    Benchmark ML models for multi-SKU forecasting
    """

    def __init__(self, feature_cols, target_col="y"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.results = []

    def train_lightgbm(self, df_train, df_val, df_test):
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(
            df_train[self.feature_cols], df_train[self.target_col],
            eval_set=[(df_val[self.feature_cols], df_val[self.target_col])],
            eval_metric="l1",
            early_stopping_rounds=50,
            verbose=False
        )

        return self.evaluate(model, df_test, "LightGBM")

    def train_xgboost(self, df_train, df_val, df_test):
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(
            df_train[self.feature_cols], df_train[self.target_col],
            eval_set=[(df_val[self.feature_cols], df_val[self.target_col])],
            verbose=False
        )

        return self.evaluate(model, df_test, "XGBoost")

    def train_random_forest(self, df_train, df_test):
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        )

        model.fit(df_train[self.feature_cols], df_train[self.target_col])

        return self.evaluate(model, df_test, "RandomForest")

    def evaluate(self, model, df_test, model_name):
        y_true = df_test[self.target_col].values
        y_pred = model.predict(df_test[self.feature_cols])

        metrics = {
            "model": model_name,
            "MAE": ForecastMetrics.mae(y_true, y_pred),
            "RMSE": ForecastMetrics.rmse(y_true, y_pred),
            "SMAPE": ForecastMetrics.smape(y_true, y_pred),
            "WAPE": ForecastMetrics.wape(y_true, y_pred)
        }

        self.results.append(metrics)
        return metrics

    def get_results(self):
        return pd.DataFrame(self.results)
