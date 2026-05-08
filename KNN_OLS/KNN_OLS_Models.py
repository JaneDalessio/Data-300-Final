#Import all necessary librarys and standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import functions from the EDA.py file
from eda import transform_data, split

def _evaluate(name, model, X_train, y_train, X_test, y_test):
    #Print and return RMSE, MAE, R² for train and test sets
    metrics = {}
    #Loops twice: once for train set, once for test set
    #Calculates all of our main metrics for the general evaluation function
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(model.predict(X_), 0, None)
        rmse  = np.sqrt(mean_squared_error(y_, preds))
        mae   = mean_absolute_error(y_, preds)
        r2    = r2_score(y_, preds)
        print(f"  {name} [{label}]  RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
        if label == "Test":
            metrics = {"RMSE": round(rmse, 1), "MAE": round(mae, 1), "R2": round(r2, 3)}
    return metrics


def run_models(X_train, X_test, y_train, y_test, tscv, feature_names):
    
    #Trains OLS and KNN models, returns dict of test-set metrics for both models
    
    results = {}

    # OLS REGRESSION 
    print("=" * 60)
    print("OLS REGRESSION")
    print("=" * 60)

    # OLS has no hyperparameters to tune, have to fit directly
    ols = Pipeline([
        ("scaler", StandardScaler()),
        ("ols",    LinearRegression())
    ])
    ols.fit(X_train, y_train)
    results["OLS"] = _evaluate("OLS", ols, X_train, y_train, X_test, y_test)

    # Coefficient plot
    coefs = ols.named_steps["ols"].coef_
    _plot_coefficients("OLS", coefs, feature_names, "#1A3A6E", "ols_coefficients.png")

    # Predicted vs actual
    _plot_predictions("OLS", ols, X_test, y_test, "#1A3A6E", "ols_predictions.png")

    # K-NEAREST NEIGHBORS
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS REGRESSOR")
    print("=" * 60)

    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsRegressor())
    ])
    knn_params = {"knn__n_neighbors": [3, 5, 7, 10, 15, 20, 25]}

    knn_search = GridSearchCV(
        knn_pipe, knn_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    knn_search.fit(X_train, y_train)
    best_k = knn_search.best_params_["knn__n_neighbors"]
    print(f"Best k: {best_k}")
    results["KNN"] = _evaluate("KNN", knn_search.best_estimator_,
                                X_train, y_train, X_test, y_test)

    # k vs RMSE sensitivity plot
    cv_res = knn_search.cv_results_
    _plot_k_sensitivity(knn_params["knn__n_neighbors"],
                        -cv_res["mean_test_score"], best_k)

    # Predicted vs actual
    _plot_predictions("KNN", knn_search.best_estimator_,
                      X_test, y_test, "#5B8DD4", "knn_predictions.png")

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df["Actual"] = y_test  # move this inside _plot_monthly_rmse's copy

    _plot_monthly_rmse({"OLS": ols, "KNN": knn_search.best_estimator_},
                    X_test_df, y_test,
                    {"OLS": "#1A3A6E", "KNN": "#5B8DD4"})

    return results


# ── Plot helpers ─────────────────────────────────────────────────────────────────

def _plot_coefficients(name, coefs, feature_names, color, filename):
    sorted_idx = np.argsort(np.abs(coefs))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in sorted_idx], coefs[sorted_idx],
            color=color, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{name} Coefficients (standardized features)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient value")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def _plot_predictions(name, model, X_test, y_test, color, filename):
    preds = np.clip(model.predict(X_test), 0, None)
    lim   = max(y_test.max(), preds.max()) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.25, s=8, color=color)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual bike count")
    ax.set_ylabel("Predicted bike count")
    ax.set_title(f"{name} — Predicted vs Actual (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def _plot_k_sensitivity(k_values, rmse_values, best_k):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, rmse_values, marker="o", color="#1A3A6E", linewidth=2)
    ax.axvline(best_k, color="#C0392B", linestyle="--", label=f"Best k={best_k}")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("CV RMSE")
    ax.set_title("KNN — Cross-Validated RMSE vs k", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("knn_k_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: knn_k_sensitivity.png")


def _plot_monthly_rmse(author, models, X_test, y_test, colors):
    """Break down test-set RMSE by month — exposes seasonal performance gap."""
    X_test = X_test.copy()
    X_test["Actual"] = y_test

    fig, ax = plt.subplots(figsize=(8, 5))
    months = sorted(X_test["month"].unique())

    for name, model in models.items():
        drop_cols = ["Actual"] + [c for c in X_test.columns if c.startswith("Pred_")]
        preds = np.clip(model.predict(X_test.drop(columns=drop_cols)), 0, None)
        X_test[f"Pred_{name}"] = preds
        monthly_rmse = [
            np.sqrt(mean_squared_error(
                X_test.loc[X_test["month"] == m, "Actual"],
                X_test.loc[X_test["month"] == m, f"Pred_{name}"]
            )) for m in months
        ]
        ax.plot(months, monthly_rmse, marker="o", label=name, color=colors[name], linewidth=2)

    ax.set_xlabel("Month (test set: Oct–Dec)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{author} Models — RMSE by Month (Test Set)", fontsize=12, fontweight="bold")
    ax.set_xticks(months)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"monthly_rmse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: monthly_rmse.png")


#Standalone run



if __name__ == "__main__":
    from sklearn.model_selection import TimeSeriesSplit
    import pandas as pd

    df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")
    df = transform_data(df)
    feature_names = [c for c in df.columns if c != "rented_bike_count"]
    tscv = TimeSeriesSplit(n_splits=5)
    X_train, X_test, y_train, y_test = split(df)

    results = run_models(X_train, X_test, y_train, y_test, tscv, feature_names)
    print("\nResults:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
