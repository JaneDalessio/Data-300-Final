"""
DATA 300 Final Project
Lasso Regression & Elastic Net Regression

Imports clean data directly from eda.py.
Returns results dict for main.py to aggregate.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Import machine learning libraries
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Import evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


## Helper function: evaluates trained model
def _evaluate(name, model, X_train, y_train, X_test, y_test):

    # Create empty dict to store results
    metrics = {}
 
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:

        # Calculate evaluation metrics
        preds = np.clip(model.predict(X_), 0, None)
        rmse  = np.sqrt(mean_squared_error(y_, preds))
        mae   = mean_absolute_error(y_, preds)
        r2    = r2_score(y_, preds)
        print(f"  {name} [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        # Only save test results
        if label == "Test":
            metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
         
    return metrics

## Main function: trains Lasso and Elastic Net Models
def run_lasso_elastic(X_train, X_test, y_train, y_test):

    feature_names = [
        "year", "month", "day", "day_of_week", "is_weekend",
        "hour", "temperature", "humidity", "wind_speed", "visibility",
        "solar_radiation", "rainfall", "snowfall", "is_holiday",
        "season_Autumn", "season_Spring", "season_Summer", "season_Winter",
        "is_peak", "is_normal", "is_low"
    ]
 
    # Extract month values for each test row
    month_idx   = feature_names.index("month")
    months_test = X_test[:, month_idx]

    # Store final results
    results = {}
    
    ## Lasso Regression Start
    print("LASSO REGRESSION")
    print("\n")

    # Create pipeline to scale data and run model
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(max_iter=10000, random_state=42))
    ])
 
    # Define possible vals for alpha
    lasso_params = {"lasso__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

    # Try alpha vals using cross validation to find best one
    lasso_search = GridSearchCV(
        lasso_pipe, lasso_params,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
 
    # Train model
    lasso_search.fit(X_train, y_train)
 
    # Get best alpha
    best_alpha_lasso = lasso_search.best_params_["lasso__alpha"]
    print(f"Best alpha: {best_alpha_lasso}")

    # Evaluate model and store results
    results["Lasso"] = _evaluate("Lasso", lasso_search.best_estimator_,
                                  X_train, y_train, X_test, y_test)

    # Get model coefficients
    lasso_coefs = lasso_search.best_estimator_.named_steps["lasso"].coef_

    # Count how many features were removed
    n_zero = int(np.sum(lasso_coefs == 0))
    print(f"Features zeroed out: {n_zero} / {len(lasso_coefs)}")

    # List which features were zero
    zeroed = [feature_names[i] for i in range(len(lasso_coefs)) if lasso_coefs[i] == 0]
    if zeroed:
        print(f"  Zeroed features: {zeroed}")

    # Display alpha performance
    _plot_alpha_sensitivity(
        "Lasso", lasso_params["lasso__alpha"],
        -lasso_search.cv_results_["mean_test_score"],
        best_alpha_lasso, "#0A7E8C", "lasso_alpha.png"
    )

    # Scatter Plot: predicted vs actual
    _plot_predictions("Lasso", lasso_search.best_estimator_,
                      X_test, y_test, "#0A7E8C", "lasso_predictions.png")


    ## Elastic Net Regression
    print("\n")
    print("ELASTIC NET REGRESSION")
    print("\n")

    # Create pipeline 
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNet(max_iter=10000, random_state=42))
    ])

    # Define parameters: alpha and l1(ridge + lass0)
    enet_params = {
        "enet__alpha":    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }
 
    # Try vals using cross validation to find best one
    enet_search = GridSearchCV(
        enet_pipe, enet_params,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # Train model
    enet_search.fit(X_train, y_train)

    # Find best vals
    best_alpha_enet = enet_search.best_params_["enet__alpha"]
    best_l1         = enet_search.best_params_["enet__l1_ratio"]
    print(f"Best alpha: {best_alpha_enet}  |  Best l1_ratio: {best_l1}")

    # Evaluate model and store results
    results["ElasticNet"] = _evaluate("ElasticNet", enet_search.best_estimator_,
                                       X_train, y_train, X_test, y_test)

    # Get model coefficients
    enet_coefs  = enet_search.best_estimator_.named_steps["enet"].coef_
    n_zero_enet = int(np.sum(enet_coefs == 0))
    print(f"Features zeroed out: {n_zero_enet} / {len(enet_coefs)}")

    # Scatter Plot: predicted vs actual
    _plot_predictions("Elastic Net", enet_search.best_estimator_,
                      X_test, y_test, "#C0392B", "enet_predictions.png")

    ## Side-by-side coefficient comparison across both models
    _plot_coefficient_comparison(
        lasso_coefs, enet_coefs, feature_names,
        best_alpha_lasso, best_alpha_enet, best_l1
    )

    ## Monthly RMSE breakdown (errors by month)
    _plot_monthly_rmse(
        {"Lasso": lasso_search.best_estimator_, "ElasticNet": enet_search.best_estimator_},
        X_test, y_test, months_test,
        {"Lasso": "#0A7E8C", "ElasticNet": "#C0392B"}
    )

    # Returns results as dict
    return results


## Plot helpers: creates the graphs

def _plot_alpha_sensitivity(name, alphas, rmse_values, best_alpha, color, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    # x-axis log scale for alpha
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
 
 
def _plot_predictions(name, model, X_test, y_test, color, filename):
    preds = np.clip(model.predict(X_test), 0, None)
    lim   = max(y_test.max(), preds.max()) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    # show actual vs predicted
    ax.scatter(y_test, preds, alpha=0.25, s=8, color=color)
    # show perfect prediction line
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual log(bike count + 1)")
    ax.set_ylabel("Predicted log(bike count + 1)")
    ax.set_title(f"{name} — Predicted vs Actual (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


## Side-by-side bar chart of Lasso vs Elastic Net coefficients
def _plot_coefficient_comparison(lasso_coefs, enet_coefs, feature_names,
                                  alpha_lasso, alpha_enet, l1_ratio):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Feature Coefficients: Lasso vs Elastic Net (standardized)",
                 fontsize=14, fontweight="bold")
 
    for ax, coefs, name, color, label in zip(
        axes,
        [lasso_coefs, enet_coefs],
        ["Lasso", "Elastic Net"],
        ["#0A7E8C", "#C0392B"],
        [f"α={alpha_lasso}", f"α={alpha_enet}, l1={l1_ratio}"]
    ):
        sorted_idx   = np.argsort(np.abs(coefs))
        sorted_coefs = coefs[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        bar_colors   = [color if c != 0 else "#CCCCCC" for c in sorted_coefs]
 
        ax.barh(sorted_names, sorted_coefs, color=bar_colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}  ({label})", fontsize=12)
        ax.set_xlabel("Coefficient value")
        ax.tick_params(labelsize=9)
 
        n_zero = int(np.sum(coefs == 0))
        ax.text(0.97, 0.03, f"{n_zero} features zeroed out",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#666666",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
 
    plt.tight_layout()
    plt.savefig("coefficient_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: coefficient_comparison.png")
 

## Calculate RMSE for each month
def _plot_monthly_rmse(models, X_test, y_test, months_test, colors):
 
    # months_test is a numpy array of month numbers for each test row
    unique_months = sorted(np.unique(months_test).astype(int))
    month_labels  = {10: "Oct", 11: "Nov", 12: "Dec"}
 
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in models.items():
        preds = np.clip(model.predict(X_test), 0, None)
        monthly_rmse = [
            np.sqrt(mean_squared_error(
                y_test[months_test == m],
                preds[months_test == m]
            )) for m in unique_months
        ]
        ax.plot(unique_months, monthly_rmse, marker="o", label=name,
                color=colors[name], linewidth=2)
 
    ax.set_xlabel("Month (test set)")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Lasso & Elastic Net — RMSE by Month (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(unique_months)
    ax.set_xticklabels([month_labels.get(m, str(m)) for m in unique_months])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("monthly_rmse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: monthly_rmse.png")


## Main Execution Function
if __name__ == "__main__":

    # Import necessary libraries
    import pandas as pd
    from eda import transform_data, split

    # Load dataset
    df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

    # Clean and transform data
    df = transform_data(df)

    # Split data into train and test
    X_train, X_test, y_train, y_test = split(df)

    # Run entire pipeline
    results = run_lasso_elastic(X_train, X_test, y_train, y_test)
    print("\nResults:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
