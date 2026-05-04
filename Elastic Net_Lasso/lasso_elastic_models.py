"""
DATA 300 Final Project
Lasso Regression & Elastic Net Regression

Imports from eda.py. Call run_lasso_elastic() with the output of split().
Returns results dict for main.py to aggregate.
 
NOTE ON METRICS:
  The target (rented_bike_count) is log-transformed in eda.py via np.log1p().
  RMSE and MAE are therefore in log-scale units, not raw bike counts.
  This is intentional — it stabilises variance and improves model fit.
  To interpret: expm1(prediction) converts back to bike count scale.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Metric helper ────────────────────────────────────────────────────────────────
 
def _evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    Prints and returns RMSE, MAE, R² for train and test sets.
    Values are in log-scale because the target was log-transformed in eda.py.
    """
    metrics = {}
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(model.predict(X_), 0, None)
        rmse  = np.sqrt(mean_squared_error(y_, preds))
        mae   = mean_absolute_error(y_, preds)
        r2    = r2_score(y_, preds)
        print(f"  {name} [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        if label == "Test":
            metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    return metrics

# ── Main function ────────────────────────────────────────────────────────────────
 
def run_lasso_elastic(X_train, X_test, y_train, y_test, tscv, feature_names, months_test):
    """
    Trains Lasso and Elastic Net models.
 
    Parameters
    ----------
    X_train, X_test    : numpy arrays from eda.split()
    y_train, y_test    : numpy arrays (log-transformed bike count)
    tscv               : TimeSeriesSplit object from eda.split()
    feature_names      : list of feature column names from eda.split()
    months_test        : numpy array of month numbers for each test row
                         (used for per-month RMSE breakdown)
 
    Returns
    -------
    results  : dict  {"Lasso": {...metrics}, "ElasticNet": {...metrics}}
    models   : dict  {"Lasso": fitted_pipeline, "ElasticNet": fitted_pipeline}
               (returned so main.py can reuse without refitting)
    """
    results = {}
    fitted  = {}
    
    # ── LASSO REGRESSION ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("LASSO REGRESSION")
    print("=" * 60)

    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(max_iter=10000, random_state=42))
    ])
    lasso_params = {"lasso__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

    lasso_search = GridSearchCV(
        lasso_pipe, lasso_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    lasso_search.fit(X_train, y_train)
    best_alpha_lasso = lasso_search.best_params_["lasso__alpha"]
    print(f"Best alpha: {best_alpha_lasso}")
    
    results["Lasso"] = _evaluate("Lasso", lasso_search.best_estimator_,
                                  X_train, y_train, X_test, y_test)
   fitted["Lasso"]  = lasso_search.best_estimator_

    lasso_coefs = lasso_search.best_estimator_.named_steps["lasso"].coef_
    n_zero      = int(np.sum(lasso_coefs == 0))
    zeroed      = [feature_names[i] for i, c in enumerate(lasso_coefs) if c == 0]
    print(f"Features zeroed out: {n_zero} / {len(lasso_coefs)}")
    if zeroed:
        print(f"  Zeroed: {zeroed}")

    # Alpha sensitivity
    _plot_alpha_sensitivity(
        "Lasso", lasso_params["lasso__alpha"],
        -lasso_search.cv_results_["mean_test_score"],
        best_alpha_lasso, "#0A7E8C", "lasso_alpha.png"
    )

    # Predicted vs actual (log scale)
    _plot_predictions("Lasso", lasso_search.best_estimator_,
                      X_test, y_test, "#0A7E8C", "lasso_predictions.png")

    # ── ELASTIC NET REGRESSION ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ELASTIC NET REGRESSION")
    print("=" * 60)

    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNet(max_iter=10000, random_state=42))
    ])
    enet_params = {
        "enet__alpha":    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    enet_search = GridSearchCV(
        enet_pipe, enet_params,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    enet_search.fit(X_train, y_train)
    best_alpha_enet = enet_search.best_params_["enet__alpha"]
    best_l1         = enet_search.best_params_["enet__l1_ratio"]
    print(f"Best alpha: {best_alpha_enet}  |  Best l1_ratio: {best_l1}")

    results["ElasticNet"] = _evaluate("ElasticNet", enet_search.best_estimator_,
                                       X_train, y_train, X_test, y_test)
    fitted["ElasticNet"]  = enet_search.best_estimator_

    enet_coefs  = enet_search.best_estimator_.named_steps["enet"].coef_
    n_zero_enet = int(np.sum(enet_coefs == 0))
    print(f"Features zeroed out: {n_zero_enet} / {len(enet_coefs)}")

    # Predicted vs actual
    _plot_predictions("Elastic Net", enet_search.best_estimator_,
                      X_test, y_test, "#C0392B", "enet_predictions.png")

    # ── Side-by-side coefficient comparison ──────────────────────────────────────
    _plot_coefficient_comparison(
        lasso_coefs, enet_coefs, feature_names,
        best_alpha_lasso, best_alpha_enet, best_l1
    )

    # ── Monthly RMSE breakdown ────────────────────────────────────────────────────
    _plot_monthly_rmse(
        {"Lasso": lasso_search.best_estimator_, "ElasticNet": enet_search.best_estimator_},
        X_test, y_test, months_test,
        {"Lasso": "#0A7E8C", "ElasticNet": "#C0392B"}
    )

    return results, fitted


# ── Plot helpers ──────────────────────────────────────────────────────────────────

def _plot_alpha_sensitivity(name, alphas, rmse_values, best_alpha, color, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(alphas, rmse_values, marker="o", color=color, linewidth=2, markersize=7)
    ax.axvline(best_alpha, color="#C0392B", linestyle="--", label=f"Best α={best_alpha}")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("CV RMSE (log scale)")
    ax.set_title(f"{name} — CV RMSE vs Alpha", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
 
 
def _plot_predictions(name, model, X_test, y_test, color, filename):
    """Predicted vs actual in log scale. Diagonal = perfect prediction."""
    preds = np.clip(model.predict(X_test), 0, None)
    lim   = max(y_test.max(), preds.max()) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.25, s=8, color=color)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual log(bike count + 1)")
    ax.set_ylabel("Predicted log(bike count + 1)")
    ax.set_title(f"{name} — Predicted vs Actual (Test Set, log scale)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
 
 
def _plot_coefficient_comparison(lasso_coefs, enet_coefs, feature_names,
                                  alpha_lasso, alpha_enet, l1_ratio):
    """Side-by-side bar chart — grey bars = zeroed out by regularization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Feature Coefficients: Lasso vs Elastic Net (standardized)",
                 fontsize=14, fontweight="bold")
 
    for ax, coefs, name, color, label in zip(
        axes,
        [lasso_coefs, enet_coefs],
        ["Lasso", "Elastic Net"],
        ["#0A7E8C", "#C0392B"],
        [f"α={alpha_lasso}", f"α={alpha_enet}, l1_ratio={l1_ratio}"]
    ):
        sorted_idx   = np.argsort(np.abs(coefs))
        sorted_coefs = coefs[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        bar_colors   = [color if c != 0 else "#CCCCCC" for c in sorted_coefs]
 
        ax.barh(sorted_names, sorted_coefs, color=bar_colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}  ({label})", fontsize=12)
        ax.set_xlabel("Coefficient value (standardized)")
        ax.tick_params(labelsize=8)
 
        n_zero = int(np.sum(coefs == 0))
        ax.text(0.97, 0.03, f"{n_zero} features zeroed out",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="#555555",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
 
    plt.tight_layout()
    plt.savefig("coefficient_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: coefficient_comparison.png")
 
 
def _plot_monthly_rmse(models, X_test, y_test, months_test, colors):
    """
    Per-month RMSE breakdown across the test set.
    Uses months_test (numpy array) from eda.split() — no DataFrame needed.
    """
    unique_months = sorted(np.unique(months_test))
    month_names   = {10: "Oct", 11: "Nov", 12: "Dec"}
 
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in models.items():
        preds = np.clip(model.predict(X_test), 0, None)
        monthly_rmse = [
            np.sqrt(mean_squared_error(
                y_test[months_test == m],
                preds[months_test == m]
            )) for m in unique_months
        ]
        ax.plot(unique_months, monthly_rmse, marker="o",
                label=name, color=colors[name], linewidth=2, markersize=7)
 
    ax.set_xlabel("Month (test set)")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Lasso & Elastic Net — RMSE by Month (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(unique_months)
    ax.set_xticklabels([month_names.get(m, str(m)) for m in unique_months])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("monthly_rmse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: monthly_rmse.png")


# ── Standalone run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from eda import transform_data, split
 
    df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")
    df = transform_data(df)
    X_train, X_test, y_train, y_test, tscv, feature_names, months_test = split(df)
 
    results, fitted = run_lasso_elastic(
        X_train, X_test, y_train, y_test, tscv, feature_names, months_test
    )
    print("\nResults:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
