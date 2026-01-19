import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.base_metrics import ForecastMetrics
from models.ml_models import BenchmarkMLModels


class SKUClustering:
    """
    Cluster SKUs based on historical demand patterns
    """

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    def build_sku_features(self, df):
        """
        Aggregate time-series statistics per SKU
        """
        sku_features = (
            df
            .groupby(["item_id", "store_id"])
            .agg(
                mean_sales=("sales", "mean"),
                std_sales=("sales", "std"),
                zero_ratio=("sales", lambda x: (x == 0).mean()),
                promo_ratio=("promo_flag", "mean")
            )
            .fillna(0)
        )

        return sku_features

    def fit(self, df):
        self.sku_features_ = self.build_sku_features(df)
        X = self.scaler.fit_transform(self.sku_features_)
        self.labels_ = self.kmeans.fit_predict(X)

        self.sku_features_["cluster"] = self.labels_
        return self.sku_features_

    def assign_clusters(self, df):
        """
        Merge cluster labels back to transactional data
        """
        return (
            df
            .merge(
                self.sku_features_[["cluster"]],
                left_on=["item_id", "store_id"],
                right_index=True,
                how="left"
            )
        )
