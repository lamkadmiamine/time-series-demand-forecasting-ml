# ==========================================================
# EDA & Feature Engineering
# Multi-SKU Demand Forecasting (Monthly)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. Monthly Aggregation & Business Rules
# ==========================================================
class MonthlyAggregator:
    """
    Objective:
    - Daily → Monthly
    - SKU / Store / Dept / Cat
    - Strategic view (trend, seasonality, promo)
    """

    def __init__(self, df_daily: pd.DataFrame):
        self.df = df_daily.copy()

    def aggregate(self):
        df_monthly = (
            self.df
            .groupby([
                "item_id", "store_id",
                "dept_id", "cat_id",
                "year", "month"
            ], as_index=False)
            .agg({
                "sales": "sum",
                "sell_price": "mean",
                "discount_pct": "mean",
                "promo_flag": "max"
            })
        )

        # Business rule:
        # sell_price & discount only exist when sales > 0
        df_monthly.loc[df_monthly["sales"] == 0, "discount_pct"] = 0

        # Sort strictly (NO leakage)
        df_monthly = df_monthly.sort_values(
            ["item_id", "store_id", "year", "month"]
        ).reset_index(drop=True)

        return df_monthly


# ==========================================================
# 2. EDA & Data Preparation
# ==========================================================
# ==========================================================
# 2. EDA & Data Preparation (avec visualisations)
# ==========================================================
class EDADataPrep:
    """
    Exploratory Data Analysis & strategic diagnostics:
    - Structure SKU / Dept / Cat
    - Sparsity
    - Seasonality
    - Trend
    - Promo & price intuition
    """

    def __init__(self, df_monthly: pd.DataFrame):
        self.df = df_monthly.copy()

    # =========================
    # Sparsity
    # =========================
    def sparsity_ratio(self):
        """
        % of zero-sales months
        """
        return (self.df["sales"] == 0).mean()

    # =========================
    # SKU Structure
    # =========================
    def plot_sku_structure(self):
        """
        SKU distribution by department & category
        """
        df = self.df

        dept_counts = (
            df[["item_id", "dept_id"]]
            .drop_duplicates()
            .groupby("dept_id")
            .size()
            .sort_values(ascending=False)
        )

        cat_counts = (
            df[["item_id", "cat_id"]]
            .drop_duplicates()
            .groupby("cat_id")
            .size()
            .sort_values(ascending=False)
        )

        n_items = df["item_id"].nunique()

        plt.figure(figsize=(15, 4))

        # Departments
        plt.subplot(1, 3, 1)
        dept_counts.plot(kind="bar")
        plt.title("SKU par département")
        plt.xlabel("Département")
        plt.ylabel("Nombre de SKU")
        plt.xticks(rotation=45)

        # Categories
        plt.subplot(1, 3, 2)
        cat_counts.plot(kind="bar")
        plt.title("SKU par catégorie")
        plt.xlabel("Catégorie")
        plt.ylabel("Nombre de SKU")

        # Summary
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.6, f"Total SKU\n{n_items}", fontsize=16)
        plt.axis("off")
        plt.title("Résumé")

        plt.tight_layout()
        plt.show()

    # =========================
    # Department-level Analysis
    # =========================
    def plot_department_diagnostics(self):
        """
        - Total sales by department
        - Seasonality Dept x Month
        - Zero-sales ratio by department
        """
        df = self.df

        sales_by_dept = (
            df.groupby("dept_id")["sales"]
            .sum()
            .sort_values()
        )

        seasonality = (
            df.groupby(["dept_id", "month"])["sales"]
            .mean()
            .unstack()
        )

        zero_ratio = (
            df.groupby("dept_id")["sales"]
            .apply(lambda x: (x == 0).mean())
        )

        plt.figure(figsize=(18, 5))

        # Total sales
        plt.subplot(1, 3, 1)
        sales_by_dept.plot(kind="barh")
        plt.title("Ventes totales par département")
        plt.xlabel("Ventes totales")
        plt.ylabel("Département")

        # Seasonality heatmap (simple)
        plt.subplot(1, 3, 2)
        plt.imshow(seasonality, aspect="auto")
        plt.colorbar(label="Ventes moyennes")
        plt.yticks(range(len(seasonality.index)), seasonality.index)
        plt.xticks(range(12), range(1, 13))
        plt.xlabel("Mois")
        plt.title("Saisonnalité Dept × Mois")

        # Zero ratio
        plt.subplot(1, 3, 3)
        zero_ratio.sort_values().plot(kind="bar")
        plt.title("Taux de zéros par département")
        plt.ylabel("Proportion de mois à 0")
        plt.xlabel("Département")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    # =========================
    # Category Time Series
    # =========================
    def plot_monthly_sales_by_category(self):
        """
        Monthly sales evolution by category
        """
        df = self.df

        monthly_cat_sales = (
            df
            .groupby(["year", "month", "cat_id"])["sales"]
            .sum()
            .reset_index()
        )

        monthly_cat_sales["date"] = pd.to_datetime(
            monthly_cat_sales["year"].astype(str) + "-" +
            monthly_cat_sales["month"].astype(str) + "-01"
        )

        pivot_monthly = (
            monthly_cat_sales
            .pivot(index="date", columns="cat_id", values="sales")
            .sort_index()
        )

        plt.figure(figsize=(14, 5))

        for cat in pivot_monthly.columns:
            plt.plot(pivot_monthly.index, pivot_monthly[cat], label=cat)

        plt.title("Évolution mensuelle des ventes par catégorie")
        plt.xlabel("Date")
        plt.ylabel("Ventes mensuelles")
        plt.legend(title="Catégorie")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # =========================
    # YoY Growth by Category
    # =========================
    def plot_yoy_growth_by_category(self):
        """
        Year-over-Year growth (%) by category
        """
        df = self.df

        annual_cat_sales = (
            df
            .groupby(["year", "cat_id"])["sales"]
            .sum()
            .reset_index()
        )

        annual_cat_sales["yoy_growth"] = (
            annual_cat_sales
            .sort_values("year")
            .groupby("cat_id")["sales"]
            .pct_change() * 100
        )

        plt.figure(figsize=(12, 5))

        for cat in annual_cat_sales["cat_id"].unique():
            data = annual_cat_sales[annual_cat_sales["cat_id"] == cat]
            plt.plot(data["year"], data["yoy_growth"], marker="o", label=cat)

            for x, y in zip(data["year"], data["yoy_growth"]):
                if not pd.isna(y):
                    plt.text(x, y, f"{y:.1f}%", fontsize=9,
                             ha="center", va="bottom")

        plt.axhline(0, color="black", linewidth=1)
        plt.title("Croissance YoY (%) par catégorie")
        plt.xlabel("Année")
        plt.ylabel("Croissance YoY (%)")
        plt.legend(title="Catégorie")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # =========================
    # Promo Effect (EDA-level)
    # =========================
    def plot_promo_effect_by_category(self):
        """
        Average sales with vs without promotion
        """
        df = self.df

        promo_sales = (
            df
            .groupby(["cat_id", "promo_flag"])["sales"]
            .mean()
            .unstack()
        )

        promo_sales.plot(kind="bar", figsize=(8, 4))
        plt.title("Ventes moyennes avec / sans promotion")
        plt.ylabel("Ventes moyennes")
        plt.xlabel("Catégorie")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()



# ==========================================================
# 3. Diagnostic Phase (Causal Intuition)
# ==========================================================
class DiagnosticTests:
    """
    Statistical diagnostics:
    - Promo impact
    - Price impact
    - Discount impact
    - Probability of selling
    - Effect by category
    """

    def __init__(self, df_monthly: pd.DataFrame):
        self.df = df_monthly.copy()
        self.df["y"] = np.log1p(self.df["sales"])

    def promo_vs_sales(self):
        with_promo = self.df[self.df["promo_flag"] == 1]["y"]
        without_promo = self.df[self.df["promo_flag"] == 0]["y"]

        stat, p_value = mannwhitneyu(
            with_promo,
            without_promo,
            alternative="greater"
        )

        print("Promo vs Sales p-value:", p_value)

        plt.figure(figsize=(8, 5))
        self.df.boxplot(column="y", by="promo_flag")
        plt.title("Impact des promotions sur log(sales)")
        plt.suptitle("")
        plt.show()

    def price_vs_sales(self):
        valid = self.df[self.df["sell_price"] > 0]
        corr, p_value = spearmanr(valid["sell_price"], valid["y"])

        print("Price Spearman corr:", corr)
        print("p-value:", p_value)

    def discount_vs_sales(self):
        corr, p_value = spearmanr(
            self.df["discount_pct"],
            self.df["y"]
        )

        print("Discount Spearman corr:", corr)
        print("p-value:", p_value)

    def probability_of_sale(self):
        df_bin = self.df.copy()
        df_bin["sold"] = (df_bin["sales"] > 0).astype(int)

        X = df_bin[["promo_flag", "discount_pct", "sell_price"]].fillna(0)
        y = df_bin["sold"]

        X_scaled = StandardScaler().fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        coef = pd.Series(model.coef_[0], index=X.columns)
        print("Logistic regression coefficients:")
        print(coef)

    def effect_by_category(self):
        effect = (
            self.df
            .groupby("cat_id")
            .apply(lambda x: spearmanr(x["discount_pct"], x["y"])[0])
            .sort_values()
        )

        return effect


# ==========================================================
# 4. Feature Engineering (Monthly – ML Safe)
# ==========================================================
class FeatureEngineeringMonthly:
    """
    Monthly feature engineering with NO leakage
    """

    def __init__(self, df_monthly: pd.DataFrame):
        self.df = df_monthly.copy()

    def prepare(self):
        df = self.df

        # Order & safety
        df = df.sort_values(
            ["item_id", "store_id", "year", "month"]
        ).reset_index(drop=True)

        assert df.duplicated(
            ["item_id", "store_id", "year", "month"]
        ).sum() == 0

        # Target & zero handling
        df["y"] = np.log1p(df["sales"])
        df["is_zero"] = (df["sales"] == 0).astype("int8")

        df["was_zero_last_month"] = (
            df
            .groupby(["item_id", "store_id"])["is_zero"]
            .shift(1)
            .fillna(0)
            .astype("int8")
        )

        # Seasonality
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Trend
        df["time_idx"] = (
            df.groupby(["item_id", "store_id"]).cumcount()
        )

        # Lags
        LAGS = [1, 3, 6, 12]
        for lag in LAGS:
            df[f"lag_{lag}"] = (
                df
                .groupby(["item_id", "store_id"])["y"]
                .shift(lag)
            )

        # Rolling stats (robust)
        df["roll_mean_3"] = (
            df
            .groupby(["item_id", "store_id"])["y"]
            .shift(1)
            .rolling(3)
            .mean()
        )

        df["roll_mean_6"] = (
            df
            .groupby(["item_id", "store_id"])["y"]
            .shift(1)
            .rolling(6)
            .mean()
        )

        df["roll_std_6"] = (
            df
            .groupby(["item_id", "store_id"])["y"]
            .shift(1)
            .rolling(6)
            .std()
        )

        # Promotions & price
        df["promo_last_3"] = (
            df
            .groupby(["item_id", "store_id"])["promo_flag"]
            .shift(1)
            .rolling(3)
            .mean()
        )

        df["price_lag_1"] = (
            df
            .groupby(["item_id", "store_id"])["sell_price"]
            .shift(1)
        )

        df["price_change_1"] = df["sell_price"] - df["price_lag_1"]

        # Category interactions (business critical)
        df["promo_foods"] = df["promo_flag"] * (df["cat_id"] == "FOODS").astype("int8")
        df["promo_hobbies"] = df["promo_flag"] * (df["cat_id"] == "HOBBIES").astype("int8")
        df["promo_household"] = df["promo_flag"] * (df["cat_id"] == "HOUSEHOLD").astype("int8")

        # Final feature set
        feature_cols = [
            "is_zero", "was_zero_last_month",
            "month_sin", "month_cos", "time_idx",
            "promo_flag", "promo_last_3",
            "sell_price", "price_lag_1", "price_change_1",
            "promo_foods", "promo_hobbies", "promo_household",
            "roll_mean_3", "roll_mean_6", "roll_std_6"
        ] + [f"lag_{l}" for l in LAGS]

        df_model = df.dropna(
            subset=feature_cols + ["y"]
        ).reset_index(drop=True)

        return df_model, feature_cols


# ==========================================================
# 5. Pipeline
# ==========================================================
class EDAMonthlyPipeline:
    """
    Full EDA + Diagnostics + Feature Engineering pipeline
    """

    def __init__(self, df_daily: pd.DataFrame):
        self.df_daily = df_daily

    def run(self):
        df_monthly = MonthlyAggregator(self.df_daily).aggregate()

        sparsity = EDADataPrep(df_monthly).sparsity_ratio()
        print("Monthly zero-sales ratio:", sparsity)

        diagnostics = DiagnosticTests(df_monthly)
        diagnostics.promo_vs_sales()
        diagnostics.price_vs_sales()
        diagnostics.discount_vs_sales()
        diagnostics.probability_of_sale()

        effect_by_cat = diagnostics.effect_by_category()
        print(effect_by_cat)

        df_model, features = FeatureEngineeringMonthly(df_monthly).prepare()

        return df_model, features
