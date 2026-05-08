"""
DATA 300 Final Project
Calls all three model files, aggregates results,
and produces the final comparison table and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error

## Import all three model modules and eda file
from eda import transform_data, split
from KNN_OLS_Models import run_KNN_OLS_models
from aidan_models import run_aidan_models
from lasso_elastic_models import run_lasso_elastic


## Color palette (consistent with individual files)
MODEL_COLORS = {
    "OLS":        "#1A3A6E",
    "KNN":        "#5B8DD4",
    "Ridge":      "#6A4BA5",
    "SGD":        "#9B7FCC",
    "Lasso":      "#0A7E8C",
    "ElasticNet": "#C0392B",
}
MONTH_NAMES = {10: "October", 11: "November", 12: "December"}

## Run EDA
def main():
    
    # show_plots=False suppresses individual EDA plots during the combined run
    # Set to True if you want to see them
    X_train, X_test, y_train, y_test, tscv, feature_names = run_eda(show_plots=False)

    # RUN ALL THREE MODEL FILES
    KNN_OLS_results      = run_KNN_OLS_models(X_train, X_test, y_train, y_test)
    aidan_results     = run_aidan_models(X_train, X_test, y_train, y_test, tscv)
    lasso_elastic_results = run_lasso_elastic_models(X_train, X_test, y_train, y_test)

    # FINAL COMPARISON TABLE
    all_results = {**KNN_OLS_results, **aidan_results, **lasso_elastic_results}

    comparison = (
        pd.DataFrame(all_results).T
        .rename(columns = {"R2": "R²"})
        .sort_values("RMSE")
        .reset_index()
        .rename(columns = {"index": "Model"})
    )

    # Assign author
    author_map = {
        "OLS": "Jane", "KNN": "Jane",
        "Ridge": "Aidan", "SGD": "Aidan",
        "Lasso": "Katherine", "ElasticNet": "Katherine"
    }
    comparison["Author"] = comparison["Model"].map(author_map)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (ranked by test-set RMSE)")
    print("=" * 60)
    print(comparison.to_string(index = False))

    comparison.to_csv("final_comparison.csv", index = False)
    print("\nSaved: final_comparison.csv")

    best = comparison.iloc[0]
    print(f"\nBest model: {best['Model']} ({best['Author']})")
    print(f"  RMSE = {best['RMSE']}  MAE = {best['MAE']}  R² = {best['R²']}")

    # COMPARISON BAR CHART (RMSE, MAE, R²)
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    fig.suptitle("All Six Models — Test Set Performance Comparison",
                 fontsize = 14, fontweight = "bold")

    for ax, metric, note in zip(
        axes,
        ["RMSE", "MAE", "R²"],
        ["lower is better", "lower is better", "higher is better"]
    ):
        bar_colors = [MODEL_COLORS[m] for m in comparison["Model"]]
        bars = ax.bar(comparison["Model"], comparison[metric],
                      color = bar_colors, width = 0.6, edgecolor = "white", linewidth = 0.5)

        # Highlight the best bar
        best_idx = (comparison[metric].idxmin() if metric != "R²"
                    else comparison[metric].idxmax())
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(f"{metric}\n({note})", fontsize = 11)
        ax.set_ylabel(metric)
        ax.tick_params(axis = "x", rotation = 30, labelsize = 9)
        ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
        fmt = "%.2f" if metric == "R²" else "%.0f"
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))

    # Legend by author
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor = "#1A3A6E", label = "OLS, KNN"),
        Patch(facecolor = "#6A4BA5", label = "Ridge, SGD"),
        Patch(facecolor = "#0A7E8C", label = "Lasso, Elastic Net"),
    ]
    fig.legend(handles = legend_elements, loc = "lower center", ncol = 3,
               bbox_to_anchor = (0.5, -0.08), fontsize = 10, frameon = False)

    plt.tight_layout(rect = [0, 0.06, 1, 1])
    plt.savefig("final_comparison_bar.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    print("Saved: final_comparison_bar.png")

    ## MONTHLY RMSE BREAKDOWN — all 6 models on the same axes
    #    Shows how performance degrades as winter approaches

    # We need the fitted models from each module — refit here using best known params
    # (In practice, save models from each file; here we refit for simplicity)
    _plot_all_monthly_rmse(all_results, X_train, X_test, y_train, y_test, comparison)

    ## RMSE HEATMAP — model × month
    #    Compact visual for poster / report
    _plot_rmse_heatmap(all_results, X_train, X_test, y_train, y_test, comparison)

    print("\nAll done — check the output files in this folder.")


## HELPERS

def _refit_all_models(X_train, y_train, tscv):
    """
    Refit each model with its best params so main.py can generate
    combined plots without passing fitted objects between files.
    Returns dict of {model_name: fitted_pipeline}.
    """
    from sklearn.linear_model import (LinearRegression, Ridge, SGDRegressor,
                                       Lasso, ElasticNet)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    models_out = {}

    # OLS
    ols = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ols.fit(X_train, y_train)
    models_out["OLS"] = ols

    # KNN
    knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
    gs = GridSearchCV(knn, {"model__n_neighbors": [3,5,7,10,15,20,25]}, 
                      scoring = "neg_root_mean_squared_error", n_jobs = -1)
    gs.fit(X_train, y_train)
    models_out["KNN"] = gs.best_estimator_

    # Ridge
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    gs = GridSearchCV(ridge, {"model__alpha": [0.01,0.1,1,5,10,50,100,500]},
                      scoring = "neg_root_mean_squared_error", n_jobs = -1)
    gs.fit(X_train, y_train)
    models_out["Ridge"] = gs.best_estimator_

    # SGD
    sgd = Pipeline([("scaler", StandardScaler()),
                    ("model", SGDRegressor(max_iter = 2000, random_state = 42))])
    gs = GridSearchCV(sgd,
                      {"model__alpha": [0.0001,0.001,0.01,0.1],
                       "model__penalty": ["l2","l1","elasticnet"]},
                      scoring = "neg_root_mean_squared_error", n_jobs = -1)
    gs.fit(X_train, y_train)
    models_out["SGD"] = gs.best_estimator_

    # Lasso
    lasso = Pipeline([("scaler", StandardScaler()),
                      ("model", Lasso(max_iter = 10000, random_state = 42))])
    gs = GridSearchCV(lasso,
                      {"model__alpha": [0.01,0.05,0.1,0.5,1,5,10,50,100]},
                      scoring = "neg_root_mean_squared_error", n_jobs = -1)
    gs.fit(X_train, y_train)
    models_out["Lasso"] = gs.best_estimator_

    # Elastic Net
    enet = Pipeline([("scaler", StandardScaler()),
                     ("model", ElasticNet(max_iter = 10000, random_state = 42))])
    gs = GridSearchCV(enet,
                      {"model__alpha": [0.01,0.05,0.1,0.5,1,5,10],
                       "model__l1_ratio": [0.1,0.3,0.5,0.7,0.9]},
                      scoring = "neg_root_mean_squared_error", n_jobs = -1)
    gs.fit(X_train, y_train)
    models_out["ElasticNet"] = gs.best_estimator_

    return models_out


def _plot_all_monthly_rmse(all_results, X_train, X_test, y_train, y_test, comparison):
    print("\nRefitting all models for combined monthly RMSE plot...")
    fitted = _refit_all_models(X_train, y_train, tscv)

    X_plot = X_test.copy()
    X_plot["Actual"] = y_test.values
    months = sorted(X_plot["Month"].unique())

    fig, ax = plt.subplots(figsize = (10, 5))
    for name, model in fitted.items():
        preds = np.clip(model.predict(X_plot.drop(columns=["Actual"])), 0, None)
        monthly_rmse = [
            np.sqrt(mean_squared_error(
                X_plot.loc[X_plot["Month"] == m, "Actual"],
                preds[X_plot["Month"].values == m]
            )) for m in months
        ]
        ax.plot(months, monthly_rmse, marker = "o", label = name,
                color = MODEL_COLORS[name], linewidth = 2, markersize = 7)

    ax.set_xlabel("Month", fontsize = 12)
    ax.set_ylabel("RMSE", fontsize = 12)
    ax.set_title("All Models — Test Set RMSE by Month (Oct–Dec)\n"
                 "Rising error reflects seasonal demand shift the model was not trained on",
                 fontsize = 12, fontweight = "bold")
    ax.set_xticks(months)
    ax.set_xticklabels([MONTH_NAMES[m] for m in months])
    ax.legend(fontsize = 9, ncol = 2)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
    plt.tight_layout()
    plt.savefig("final_monthly_rmse_all.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    print("Saved: final_monthly_rmse_all.png")


def _plot_rmse_heatmap(all_results, X_train, X_test, y_train, y_test, comparison):
    fitted = _refit_all_models(X_train, y_train, tscv)
    X_plot = X_test.copy()
    X_plot["Actual"] = y_test.values
    months = sorted(X_plot["Month"].unique())
    models = list(comparison["Model"])

    # Build RMSE matrix
    matrix = []
    for name in models:
        model = fitted[name]
        preds = np.clip(model.predict(X_plot.drop(columns = ["Actual"])), 0, None)
        row = [
            np.sqrt(mean_squared_error(
                X_plot.loc[X_plot["Month"] == m, "Actual"],
                preds[X_plot["Month"].values == m]
            )) for m in months
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize = (7, 5))
    im = ax.imshow(matrix, cmap = "RdYlGn_r", aspect = "auto")
    plt.colorbar(im, ax = ax, label = "RMSE")

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([MONTH_NAMES[m] for m in months])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("RMSE Heatmap — Model × Month (Test Set)\nGreen = lower error, Red = higher error",
                 fontsize = 12, fontweight = "bold")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(months)):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha = "center", va = "center",
                    fontsize = 10, color = "white" if matrix[i,j] > matrix.mean() else "black",
                    fontweight = "bold")

    plt.tight_layout()
    plt.savefig("final_rmse_heatmap.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    print("Saved: final_rmse_heatmap.png")


# ── Run ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
