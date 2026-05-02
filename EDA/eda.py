"""
DATA 300 Final Project
eda.py — Exploratory Data Analysis
Author: Aidan MacIntosh

This file handles ALL data loading, cleaning, exploration, and splitting.
Every other file imports run_eda() from here and receives clean, ready-to-use data.

Returns:
    X_train, X_test, y_train, y_test  — chronological 75/25 split
    tscv                               — TimeSeriesSplit(n_splits=5) for CV
    feature_names                      — list of feature column names
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ── Palette ─────────────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#0A7E8C", "#6A4BA5", "#F59E0B", "#C0392B"]
SEASON_COLORS  = {1: "#97BC62", 2: "#F59E0B", 3: "#C0392B", 4: "#1A3A6E"}
SEASON_NAMES   = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — called by all model files
# ════════════════════════════════════════════════════════════════════════════════

def run_eda(filepath="SeoulBikeData.csv", show_plots=True):
    """
    Loads and cleans the Seoul Bike dataset, runs full EDA,
    and returns a clean chronological train/test split.

    Parameters
    ----------
    filepath   : path to SeoulBikeData.csv
    show_plots : set False to suppress plots (useful when called from main.py)

    Returns
    -------
    X_train, X_test, y_train, y_test, tscv, feature_names
    """

    print("=" * 60)
    print("EDA — Seoul Bike Sharing Demand")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath, encoding="unicode_escape")
    df.columns = [
        "Date", "BikeCount", "Hour", "Temperature", "Humidity",
        "WindSpeed", "Visibility", "DewPoint", "SolarRadiation",
        "Rainfall", "Snowfall", "Seasons", "Holiday", "FunctioningDay"
    ]
    print(f"\nRaw shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}\n")

    # ── 2. Clean ─────────────────────────────────────────────────────────────────
    # Drop non-functioning days — system was closed, counts are 0 and uninformative
    df = df[df["FunctioningDay"] == "Yes"].copy()

    # Parse and sort by date (critical for time-series integrity)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)

    # ── 3. EDA on raw features ───────────────────────────────────────────────────
    _plot_outcome_distribution(df, show_plots)
    _plot_outlier_detection(df, show_plots)
    _plot_correlation_heatmap(df, show_plots)
    _plot_hourly_demand(df, show_plots)
    _plot_seasonal_demand(df, show_plots)

    # ── 4. Feature engineering ───────────────────────────────────────────────────
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

    # ── 5. Drop Dew Point — highly correlated with Temperature (r = 0.91) ────────
    print("Dropping DewPoint (r=0.91 with Temperature — multicollinearity)\n")
    df.drop(columns=["DewPoint"], inplace=True)

    # ── 6. Encode categoricals ───────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["Seasons", "Holiday"], drop_first=True)
    df.drop(columns=["Date", "FunctioningDay", "DayOfWeek"], inplace=True)

    # ── 7. K-Means clustering (before splitting — uses full dataset) ─────────────
    _run_clustering(df, show_plots)

    # ── 8. Chronological train/test split ────────────────────────────────────────
    # 75% train = Jan–Sep, 25% test = Oct–Dec
    # Strict chronological order — no shuffling
    y = df["BikeCount"]
    X = df.drop(columns=["BikeCount"])

    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    tscv = TimeSeriesSplit(n_splits=5)
    feature_names = list(X.columns)

    print("─" * 60)
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print("─" * 60 + "\n")

    return X_train, X_test, y_train, y_test, tscv, feature_names


# ════════════════════════════════════════════════════════════════════════════════
# EDA PLOTS
# ════════════════════════════════════════════════════════════════════════════════

def _plot_outcome_distribution(df, show):
    """KDE of BikeCount — shows right skew and leptokurtic shape."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df["BikeCount"].plot.kde(ax=ax, color="#0A7E8C", linewidth=2)
    ax.set_xlabel("Rented Bike Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Kernel Density Estimate — Rented Bike Count", fontsize=13, fontweight="bold")
    ax.axvline(df["BikeCount"].mean(),   color="#C0392B", linestyle="--", label=f"Mean: {df['BikeCount'].mean():.0f}")
    ax.axvline(df["BikeCount"].median(), color="#F59E0B", linestyle="--", label=f"Median: {df['BikeCount'].median():.0f}")
    ax.legend()
    skew = df["BikeCount"].skew()
    kurt = df["BikeCount"].kurtosis()
    ax.text(0.97, 0.95, f"Skew: {skew:.2f}\nKurtosis: {kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=10)
    plt.tight_layout()
    plt.savefig("eda_outcome_kde.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print(f"Outcome — Mean: {df['BikeCount'].mean():.1f}, Median: {df['BikeCount'].median():.1f}, "
          f"Skew: {skew:.3f}, Kurtosis: {kurt:.3f}")
    print("Saved: eda_outcome_kde.png")


def _plot_outlier_detection(df, show):
    """Boxplot of BikeCount with IQR fence annotation."""
    Q1, Q3 = df["BikeCount"].quantile(0.25), df["BikeCount"].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR
    outliers = df[df["BikeCount"] > upper_fence]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(df["BikeCount"], vert=False, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#0A7E8C")
    bp["boxes"][0].set_alpha(0.8)
    ax.axvline(upper_fence, color="#C0392B", linestyle="--",
               label=f"Upper fence: {upper_fence:.0f}")
    ax.set_xlabel("Rented Bike Count", fontsize=12)
    ax.set_title("Outlier Detection — Rented Bike Count", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("eda_outlier_boxplot.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()

    print(f"\nOutlier detection — Upper fence: {upper_fence:.1f}, Lower fence: 0")
    print(f"Total outliers: {len(outliers)} ({100*len(outliers)/len(df):.1f}% of data)")
    print(f"Outlier hours: {sorted(outliers['Hour'].unique())}")
    print("Saved: eda_outlier_boxplot.png")


def _plot_correlation_heatmap(df, show):
    """Absolute correlation heatmap — highlights DewPoint/Temperature collinearity."""
    numeric = df.select_dtypes(include="number").drop(columns=["BikeCount"], errors="ignore")
    # Include BikeCount for full picture
    cols = ["BikeCount"] + [c for c in numeric.columns if c != "BikeCount"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)

    # Annotate each cell
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(val) > 0.6 else "black")

    ax.set_title("Correlation Heatmap — Seoul Bike Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()

    # Print high-correlation pairs
    abs_corr = corr.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    high = [(r, c, round(corr.loc[r, c], 4))
            for r in upper.index for c in upper.columns
            if upper.loc[r, c] > 0.70]
    print("\nHighly correlated pairs (|r| > 0.70):")
    for a, b, v in sorted(high, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {a} ↔ {b}: r = {v}")
    print("Saved: eda_correlation_heatmap.png")


def _plot_hourly_demand(df, show):
    """Mean bike demand by hour of day — shows bimodal commute peaks."""
    hourly = df.groupby("Hour")["BikeCount"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(hourly.index, hourly.values, color="#0A7E8C", alpha=0.85, width=0.7)

    # Highlight peak hours
    peak_hours = [8, 18]
    for h in peak_hours:
        bars[h].set_color("#C0392B")
        bars[h].set_alpha(1.0)

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Mean Rented Bike Count", fontsize=12)
    ax.set_title("Average Bike Demand by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xticks(range(24))
    ax.annotate("Morning\ncommute", xy=(8, hourly[8]), xytext=(10, hourly[8]+50),
                arrowprops=dict(arrowstyle="->", color="#C0392B"), color="#C0392B", fontsize=9)
    ax.annotate("Evening\ncommute", xy=(18, hourly[18]), xytext=(20, hourly[18]+50),
                arrowprops=dict(arrowstyle="->", color="#C0392B"), color="#C0392B", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("eda_hourly_demand.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print("\nSaved: eda_hourly_demand.png")


def _plot_seasonal_demand(df, show):
    """Boxplot of BikeCount by season — shows temperature-driven seasonal variation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    season_data = [df[df["Seasons"] == s]["BikeCount"].values for s in [1, 2, 3, 4]]
    bp = ax.boxplot(season_data, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, s in zip(bp["boxes"], [1, 2, 3, 4]):
        patch.set_facecolor(SEASON_COLORS[s])
        patch.set_alpha(0.85)
    ax.set_xticklabels([SEASON_NAMES[s] for s in [1, 2, 3, 4]], fontsize=11)
    ax.set_ylabel("Rented Bike Count", fontsize=12)
    ax.set_title("Bike Demand Distribution by Season", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("eda_seasonal_demand.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print("Saved: eda_seasonal_demand.png")

    print("\nSeasonal means:")
    for s in [1, 2, 3, 4]:
        mean = df[df["Seasons"] == s]["BikeCount"].mean()
        print(f"  {SEASON_NAMES[s]}: {mean:.0f} bikes/hr avg")


# ════════════════════════════════════════════════════════════════════════════════
# K-MEANS CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════

def _run_clustering(df, show):
    """
    K-Means clustering (k=4) on demand-relevant features.
    Validates clusters against Season and Hour of Day using PCA.
    Produces: elbow curve, PCA scatter (3-panel), demand boxplot.
    """
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)

    cluster_features = ["BikeCount", "Temperature", "Hour", "Humidity", "SolarRadiation"]
    X_clust = df[cluster_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    # ── Elbow method ─────────────────────────────────────────────────────────────
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", color="#0A7E8C", linewidth=2, markersize=7)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method — Optimal k for K-Means", fontsize=13, fontweight="bold")
    ax.set_xticks(list(k_range))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axvline(4, color="#C0392B", linestyle="--", linewidth=1.5, label="k=4 (chosen)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("eda_clustering_elbow.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print("Saved: eda_clustering_elbow.png")

    # ── Fit k=4 ──────────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    print("\nCluster sizes:")
    for c, n in zip(*np.unique(labels, return_counts=True)):
        print(f"  Cluster {c}: {n} obs ({100*n/len(labels):.1f}%)")

    # ── PCA for 2D visualization ──────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%} (total={sum(var):.1%})")

    df_plot = df.loc[X_clust.index].copy()
    df_plot["Cluster"] = labels
    df_plot["PC1"]     = X_pca[:, 0]
    df_plot["PC2"]     = X_pca[:, 1]

    # Subsample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df_plot), size=min(8000, len(df_plot)), replace=False)
    dp  = df_plot.iloc[idx]

    # ── Three-panel PCA scatter ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("K-Means (k=4) Validated Against Season & Hour of Day",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel 1 — cluster labels
    ax = axes[0]
    for c in range(4):
        mask = dp["Cluster"] == c
        ax.scatter(dp.loc[mask, "PC1"], dp.loc[mask, "PC2"],
                   s=8, alpha=0.4, color=CLUSTER_COLORS[c], label=f"Cluster {c}")
    ax.set_title("K-Means Clusters", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=9)

    # Panel 2 — season overlay
    ax = axes[1]
    for s, color in SEASON_COLORS.items():
        mask = dp["Seasons"] == s
        if mask.any():
            ax.scatter(dp.loc[mask, "PC1"], dp.loc[mask, "PC2"],
                       s=8, alpha=0.4, color=color, label=SEASON_NAMES[s])
    ax.set_title("Season Overlay\n(Do clusters align with seasons?)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=9)

    # Panel 3 — hour overlay
    ax = axes[2]
    sc = ax.scatter(dp["PC1"], dp["PC2"], c=dp["Hour"],
                    cmap="plasma", s=8, alpha=0.4, vmin=0, vmax=23)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Hour of Day", fontsize=9)
    cbar.set_ticks([0, 6, 12, 18, 23])
    ax.set_title("Hour of Day Overlay\n(Do clusters align with time?)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")

    plt.tight_layout()
    plt.savefig("eda_clustering_pca.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print("Saved: eda_clustering_pca.png")

    # ── Demand boxplot by cluster ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_cluster = [df_plot[df_plot["Cluster"] == c]["BikeCount"].values for c in range(4)]
    bp = ax.boxplot(data_by_cluster, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], CLUSTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticklabels([f"Cluster {c}" for c in range(4)], fontsize=11)
    ax.set_ylabel("Rented Bike Count", fontsize=12)
    ax.set_title("Bike Demand Distribution by K-Means Cluster",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("eda_clustering_boxplot.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    print("Saved: eda_clustering_boxplot.png")

    # ── Cluster profile ───────────────────────────────────────────────────────────
    profile = (df_plot.groupby("Cluster")
               .agg(
                   BikeCount_mean  = ("BikeCount",      "mean"),
                   Temp_mean       = ("Temperature",    "mean"),
                   Hour_mean       = ("Hour",           "mean"),
                   Humidity_mean   = ("Humidity",       "mean"),
                   DominantSeason  = ("Seasons", lambda x: SEASON_NAMES[x.mode()[0]])
               )
               .round(1))
    print("\nCluster profiles:")
    print(profile.to_string())
    print()


# ════════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tscv, feature_names = run_eda()
    print("EDA complete. Data is ready for modeling.")
