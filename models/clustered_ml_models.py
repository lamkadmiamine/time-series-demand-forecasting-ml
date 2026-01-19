import pandas as pd

from models.ml_models import BenchmarkMLModels


class ClusteredMLForecast:
    """
    Train one ML model per SKU cluster
    """

    def __init__(self, feature_cols, target_col="y"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.results = []

    def train_per_cluster(self, df, cluster_col="cluster"):
        clusters = df[cluster_col].unique()

        for cluster in clusters:
            df_cluster = df[df[cluster_col] == cluster]

            if len(df_cluster) < 100:
                continue

            df_train, df_val, df_test = self.temporal_split(df_cluster)

            ml = BenchmarkMLModels(self.feature_cols, self.target_col)
            metrics = ml.train_lightgbm(df_train, df_val, df_test)

            metrics["cluster"] = int(cluster)
            self.results.append(metrics)

        return pd.DataFrame(self.results)

    @staticmethod
    def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            df.iloc[:train_end],
            df.iloc[train_end:val_end],
            df.iloc[val_end:]
        )
