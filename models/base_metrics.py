import numpy as np

class ForecastMetrics:
    """Standard forecasting metrics"""

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def smape(y_true, y_pred):
        return np.mean(
            2 * np.abs(y_pred - y_true) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        ) * 100

    @staticmethod
    def wape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

