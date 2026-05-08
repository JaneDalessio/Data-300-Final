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


## Main function: trains Lasso and Elastic Net Models
def run_lasso_elastic(X_train, X_test, y_train, y_test):

    feature_names = [
        "year", "month", "day", "day_of_week", "is_weekend",
        "hour", "temperature", "humidity", "wind_speed",
        "visibility", "solar_radiation", "rainfall", "snowfall", "is_holiday",
        "season_Autumn", "season_Spring", "season_Summer", "season_Winter",
        "rbc_1h", "rbc_2h", "rbc_3h", "rbc_24h", "rbc_48h", "rbc_168h",
        "rbc_1h_m", "rbc_2h_m", "rbc_3h_m", "rbc_24h_m", "rbc_48h_m",
        "rbc_168h_m", "rbc_mean_3h", "rbc_std_3h", "rbc_mean_6h",
        "rbc_mean_12h", "rbc_mean_24h", "rbc_std_24h", "rbc_mean_48h",
        "rbc_mean_168h", "rbc_mean_3h_m", "rbc_std_3h_m", "rbc_mean_6h_m",
        "rbc_mean_12h_m", "rbc_mean_24h_m", "rbc_std_24h_m", "rbc_mean_48h_m",
        "rbc_mean_168h_m", "is_peak", "is_normal", "is_low",
        ]

    # Store final results
    results = {}

    ## Lasso Regression
    print("LASSO REGRESSION")
    print("\n")

    # Create pipeline to scale data and run model
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(max_iter = 10000, random_state = 42))
    ])

    # Define possible vals for alpha
    lasso_params = {"lasso__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

    # Try alpha vals using cross validation to find best one
    lasso_search = GridSearchCV(
        lasso_pipe, lasso_params,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    # Train model
    lasso_search.fit(X_train, y_train)

    # Get best alpha
    best_alpha_lasso = lasso_search.best_params_["lasso__alpha"]
    print(f"Best alpha: {best_alpha_lasso}")

    # Evaluate Lasso on train and test sets
    lasso_metrics = {}
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(lasso_search.best_estimator_.predict(X_), 0, None)
        rmse = np.sqrt(mean_squared_error(y_, preds))
        mae = mean_absolute_error(y_, preds)
        r2 = r2_score(y_, preds)
        print(f"  Lasso [{label}]  RMSE = {rmse:.4f}  MAE = {mae:.4f}  R² = {r2:.4f}")
        if label == "Test":
            lasso_metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    results["Lasso"] = lasso_metrics

    # Get model coefficients
    lasso_coefs = lasso_search.best_estimator_.named_steps["lasso"].coef_

    # Count how many features were removed
    n_zero = int(np.sum(lasso_coefs == 0))
    print(f"Features zeroed out: {n_zero} / {len(lasso_coefs)}")

    # List which features were zeroed
    zeroed = [feature_names[i] for i in range(len(lasso_coefs)) if lasso_coefs[i] == 0]
    if zeroed:
        print(f"  Zeroed features: {zeroed}")


    ## Elastic Net Regression
    print("\n")
    print("ELASTIC NET REGRESSION")
    print("\n")

    # Create pipeline
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(max_iter = 10000, random_state = 42))
    ])

    # Define parameters: alpha and l1_ratio (ridge + lasso blend)
    enet_params = {
        "enet__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Try vals using cross validation to find best one
    enet_search = GridSearchCV(
        enet_pipe, enet_params,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    # Train model
    enet_search.fit(X_train, y_train)

    # Find best vals
    best_alpha_enet = enet_search.best_params_["enet__alpha"]
    best_l1 = enet_search.best_params_["enet__l1_ratio"]
    print(f"Best alpha: {best_alpha_enet}  |  Best l1_ratio: {best_l1}")

    # Evaluate Elastic Net on train and test sets
    enet_metrics = {}
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(enet_search.best_estimator_.predict(X_), 0, None)
        rmse = np.sqrt(mean_squared_error(y_, preds))
        mae = mean_absolute_error(y_, preds)
        r2 = r2_score(y_, preds)
        print(f"  ElasticNet [{label}]  RMSE = {rmse:.4f}  MAE = {mae:.4f}  R² = {r2:.4f}")
        if label == "Test":
            enet_metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    results["ElasticNet"] = enet_metrics

    # Get model coefficients
    enet_coefs = enet_search.best_estimator_.named_steps["enet"].coef_
    n_zero_enet = int(np.sum(enet_coefs == 0))
    print(f"Features zeroed out: {n_zero_enet} / {len(enet_coefs)}")


    ## Side-by-side coefficient comparison across both models
    fig, axes = plt.subplots(1, 2, figsize = (14, 6), sharey = True)
    fig.suptitle("Feature Coefficients: Lasso vs Elastic Net (standardized)",
                 fontsize = 14, fontweight = "bold")

    for ax, coefs, name, color, label in zip(
        axes,
        [lasso_coefs, enet_coefs],
        ["Lasso", "Elastic Net"],
        ["#0A7E8C", "#C0392B"],
        [f"α = {best_alpha_lasso}", f"α = {best_alpha_enet}, l1 = {best_l1}"]
    ):
        sorted_idx = np.argsort(np.abs(coefs))
        sorted_coefs = coefs[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        bar_colors = [color if c != 0 else "#CCCCCC" for c in sorted_coefs]

        ax.barh(sorted_names, sorted_coefs, color = bar_colors, alpha = 0.85)
        ax.axvline(0, color = "black", linewidth = 0.8, linestyle = "--")
        ax.set_title(f"{name}  ({label})", fontsize = 12)
        ax.set_xlabel("Coefficient value")
        ax.tick_params(labelsize = 9)

        n_zero = int(np.sum(coefs == 0))
        ax.text(0.97, 0.03, f"{n_zero} features zeroed out",
                transform = ax.transAxes, ha = "right", va = "bottom",
                fontsize = 9, color = "#666666",
                bbox = dict(boxstyle = "round", facecolor = "white", alpha = 0.7))

    plt.tight_layout()
    plt.savefig("coefficient_comparison.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    print("Saved: coefficient_comparison.png")

    # Return results as dict
    return results


## Main Execution Function
if __name__ == "__main__":

    # Import necessary libraries
    import pandas as pd
    from eda import transform_data, split

    # Load dataset
    df = pd.read_csv("SeoulBikeData.csv", encoding = "unicode_escape")

    # Clean and transform data
    df = transform_data(df)

    # Split data into train and test
    X_train, X_test, y_train, y_test = split(df)

    # Run entire pipeline
    results = run_lasso_elastic(X_train, X_test, y_train, y_test)
    print("\nResults:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
