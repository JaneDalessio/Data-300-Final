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


## Metrics heatmap
# Green = good performance, Red = poor performance

metrics_df = comparison.set_index("Model")[["RMSE", "MAE", "R²"]].astype(float)

# Normalize each column to 0-1 so the colors are meaningful across metrics
# For RMSE and MAE: lower is better, so we invert (1 - normalized)
# For R²: higher is better, so we keep as is
normalized = metrics_df.copy()
for col in ["RMSE", "MAE"]:
    col_min = metrics_df[col].min()
    col_max = metrics_df[col].max()
    normalized[col] = 1 - (metrics_df[col] - col_min) / (col_max - col_min)
r2_min = metrics_df["R²"].min()
r2_max = metrics_df["R²"].max()
normalized["R²"] = (metrics_df["R²"] - r2_min) / (r2_max - r2_min)

fig, ax = plt.subplots(figsize = (8, 5))
im = ax.imshow(normalized.values, cmap = "RdYlGn", aspect = "auto", vmin = 0, vmax = 1)

ax.set_xticks(range(len(metrics_df.columns)))
ax.set_xticklabels(metrics_df.columns, fontsize = 11)
ax.set_yticks(range(len(metrics_df.index)))
ax.set_yticklabels(metrics_df.index, fontsize = 10)
ax.set_title("Model Performance Heatmap\n(Green = better, Red = worse)",
             fontsize = 12, fontweight = "bold")

# Annotate each cell with the actual metric value
for i in range(len(metrics_df.index)):
    for j in range(len(metrics_df.columns)):
        val = metrics_df.iloc[i, j]
        ax.text(j, i, f"{val:.4f}", ha = "center", va = "center",
                fontsize = 10, fontweight = "bold",
                color = "black")

plt.tight_layout()
plt.savefig("final_heatmap.png", dpi = 150, bbox_inches = "tight")
plt.show()
print("Saved: final_heatmap.png")
