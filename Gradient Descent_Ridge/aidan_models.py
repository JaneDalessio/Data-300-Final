"""
DATA 300 Final Project
aidan_models.py — Ridge Regression & SGD Regression
Author: Aidan MacIntosh

Imports clean data directly from eda.py.
Returns results dict for main.py to aggregate.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from eda import run_eda


def _evaluate(name, model, X_train, y_train, X_test, y_test):
    metrics = {}
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(model.predict(X_), 0, None)
        rmse  = np.sqrt(mean_squared_error(y_, preds))
        mae   = mean_absolute_error(y_, preds)
        r2    = r2_score(y_, preds)
        print(f"  {name} [{label}]  RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
        if label == "Test":
            metrics = {"RMSE": round(rmse, 1), "MAE": round(mae, 1), "R2": round(r2, 3)}
    return metrics


def run_aidan_models(X_train, X_test, y_train, y_test, tscv, feature_names):
    """
    Trains Ridge and SGD models.
    Returns dict of test-set metrics for both models.
    """
    results = {}

    # ── RIDGE REGRESSION ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("RIDGE REGRESSION (Aidan)")
    print("=" * 60)

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge())
    ])
    ridge_params = {"ridge__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]}

    ridge_search = GridSearchCV(
        ridge_pipe, ridge_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    ridge_search.fit(X_train, y_train)
    best_alpha = ridge_search.best_params_["ridge__alpha"]
    print(f"Best alpha: {best_alpha}")
    results["Ridge"] = _evaluate("Ridge", ridge_search.best_estimator_,
                                  X_train, y_train, X_test, y_test)

    # Alpha sensitivity plot
    _plot_alpha_sensitivity(
        "Ridge", ridge_params["ridge__alpha"],
        -ridge_search.cv_results_["mean_test_score"],
        best_alpha, "#6A4BA5", "aidan_ridge_alpha.png"
    )

    # Coefficient plot
    ridge_coefs = ridge_search.best_estimator_.named_steps["ridge"].coef_
    _plot_coefficients("Ridge", ridge_coefs, feature_names, "#6A4BA5", "aidan_ridge_coefficients.png")

    # Predicted vs actual
    _plot_predictions("Ridge", ridge_search.best_estimator_,
                      X_test, y_test, "#6A4BA5", "aidan_ridge_predictions.png")

    # ── SGD REGRESSION ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SGD REGRESSION (Aidan)")
    print("=" * 60)

    sgd_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd",    SGDRegressor(max_iter=2000, random_state=42))
    ])
    sgd_params = {
        "sgd__alpha":   [0.0001, 0.001, 0.01, 0.1],
        "sgd__penalty": ["l2", "l1", "elasticnet"]
    }

    sgd_search = GridSearchCV(
        sgd_pipe, sgd_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    sgd_search.fit(X_train, y_train)
    print(f"Best params: {sgd_search.best_params_}")
    results["SGD"] = _evaluate("SGD", sgd_search.best_estimator_,
                                X_train, y_train, X_test, y_test)

    # Predicted vs actual
    _plot_predictions("SGD", sgd_search.best_estimator_,
                      X_test, y_test, "#9B7FCC", "aidan_sgd_predictions.png")

    # Monthly RMSE breakdown
    _plot_monthly_rmse(
        "Aidan",
        {"Ridge": ridge_search.best_estimator_, "SGD": sgd_search.best_estimator_},
        X_test, y_test,
        {"Ridge": "#6A4BA5", "SGD": "#9B7FCC"}
    )

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────────

def _plot_alpha_sensitivity(name, alphas, rmse_values, best_alpha, color, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(alphas, rmse_values, marker="o", color=color, linewidth=2, markersize=7)
    ax.axvline(best_alpha, color="#C0392B", linestyle="--", label=f"Best α={best_alpha}")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("CV RMSE")
    ax.set_title(f"{name} — CV RMSE vs Alpha", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


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


def _plot_monthly_rmse(author, models, X_test, y_test, colors):
    X_test = X_test.copy()
    X_test["Actual"] = y_test.values
    months = sorted(X_test["Month"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in models.items():
        preds = np.clip(model.predict(X_test.drop(columns=["Actual"])), 0, None)
        X_test[f"Pred_{name}"] = preds
        monthly_rmse = [
            np.sqrt(mean_squared_error(
                X_test.loc[X_test["Month"] == m, "Actual"],
                X_test.loc[X_test["Month"] == m, f"Pred_{name}"]
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
    plt.savefig("aidan_monthly_rmse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: aidan_monthly_rmse.png")


# ── Standalone run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tscv, feature_names = run_eda(show_plots=False)
    results = run_aidan_models(X_train, X_test, y_train, y_test, tscv, feature_names)
    print("\nAidan's results:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
