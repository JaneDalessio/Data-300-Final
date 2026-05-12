# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import EDA functions
from eda import transform_data, split

# Import model functions
from lasso_elastic_models import run_lasso_elastic
from KNN_OLS_Models import run_KNN_OLS_models
from sgd_ridge_models import run_sgd_ridge


# Load and prepare data
df = pd.read_csv("SeoulBikeData.csv", encoding = "unicode_escape")
df = transform_data(df)
X_train, X_test, y_train, y_test = split(df)

# Run all models
print("RUNNING ALL MODELS")
print("\n")

# Store all results
lasso_elastic_results = run_lasso_elastic(X_train, X_test, y_train, y_test)
knn_ols_results = run_KNN_OLS_models(X_train, X_test, y_train, y_test)
sgd_ridge_results = run_sgd_ridge(X_train, X_test, y_train, y_test)

# Combine all results
all_results = {**knn_ols_results, **sgd_ridge_results, **lasso_elastic_results}


## Final comparison table 
comparison = (
    pd.DataFrame(all_results)
    .T
    .rename(columns = {"R2": "R²"})
    .sort_values("RMSE")
    .reset_index()
    .rename(columns = {"index": "Model"})
)
print("Final Model Comparison (ranked by test-set RMSE)")
print(comparison.to_string(index = False))
print("\n")

comparison.to_csv("final_comparison.csv", index = False)
print("Saved: final_comparison.csv")


## Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize = (15, 5))
fig.suptitle("All Models — Test Set Performance Comparison",
             fontsize = 14, fontweight = "bold")

for ax, metric, note in zip(
    axes,
    ["RMSE", "MAE", "R²"],
    ["lower is better", "lower is better", "higher is better"]
):
    ax.bar(comparison["Model"], comparison[metric], width = 0.6)
    ax.set_title(f"{metric} ({note})", fontsize = 11)
    ax.set_ylabel(metric)
    ax.tick_params(axis = "x", rotation = 30, labelsize = 9)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.4)

plt.tight_layout()
plt.savefig("final_comparison.png", dpi = 150, bbox_inches = "tight")
plt.show()
print("Saved: final_comparison.png")


