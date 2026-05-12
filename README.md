# Assessing Regression Models for Bike Share Demand in Seoul

**DATA 300 — Statistical and Machine Learning | Spring 2026**  
Jane Dalessio · Aidan MacIntosh · Katherine Robles  
Professor: Lulu Wang

---

## Overview

This project compares six regression models for predicting hourly bike-sharing demand in Seoul, South Korea using the [Seoul Bike Sharing Demand dataset](https://doi.org/10.24432/C5F62R) from the UCI Machine Learning Repository. The goal is to identify which regression approach best predicts the number of bikes rented in a given hour, evaluated using RMSE, MAE, and R².

**Models compared:** OLS · KNN · Ridge · SGD · Lasso · Elastic Net

---

## Repository Structure

```
Data-300-Final/
│
├── EDA/                        ← Start here
│   └── eda.py
│
├── Model Results/              ← Final results
│   ├── models.py
│   └── results.ipynb
│
├── Elastic Net_Lasso/          ← Work in progress / proof of work
├── Gradient Descent_Ridge/     ← Work in progress / proof of work
├── KNN_OLS/                    ← Work in progress / proof of work
│
```

> **Note:** The `Elastic Net_Lasso`, `Gradient Descent_Ridge`, and `KNN_OLS` folders contain earlier development work and iterations from throughout the project. They are kept as proof of work but are **not** needed to reproduce the final results. To see the final analysis, refer to the `EDA` and `Model Results` folders only.

---

## Final Files

### `EDA/eda.py`

Contains all data loading, cleaning, and preprocessing logic. Two functions are exported:

- **`transform_data(df)`** — Takes the raw CSV as a DataFrame and applies all transformations: renames columns, extracts temporal features (month, day, day-of-week, weekend flag), one-hot encodes seasons, drops dew point temperature due to high multicollinearity with temperature (r = 0.91), removes non-functioning days, log-transforms the bike count outcome to address right skew, adds lag features at 1h/2h/3h/24h/48h/168h, adds rolling mean and standard deviation windows, and creates demand group flags (peak, normal, low hours).

- **`split(df)`** — Applies a strict chronological 75/25 train/test split with no shuffling. The training set covers approximately January–September and the test set covers October–December, preserving the temporal order of the data.

Also contains visualization functions used during EDA: `rbc_kde_plot`, `rbc_boxplot`, `corr_matrix`, and `demand_bar_plot`.

---

### `Model Results/models.py`

Contains training functions for all six regression models. Each function fits a `StandardScaler → model` pipeline and returns the best fitted model along with its hyperparameters. Hyperparameter tuning is done via `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` cross-validation.

- **`train_ols(X_train, Y_train)`** — Ordinary Least Squares. No hyperparameters to tune.
- **`train_knn(X_train, Y_train)`** — K-Nearest Neighbors. Searches over k ∈ {3, 5, 7}.
- **`train_ridge(X_train, Y_train)`** — Ridge Regression. Searches over α ∈ {0.01 … 500}.
- **`train_sgd(X_train, Y_train)`** — SGD Regressor. Searches over α and penalty type.
- **`train_lasso(X_train, Y_train)`** — Lasso Regression. Searches over α ∈ {0.01 … 100}.
- **`train_elastic_net(X_train, Y_train)`** — Elastic Net. Searches over α and l1_ratio.

Also contains:
- **`get_performance(model, X_train, X_test, Y_train, Y_test)`** — Returns a dict of RMSE, MAE, and R² for both train and test sets.
- **`clean_print_metrics(metrics)`** — Prints formatted train/test metrics to the console.

---

### `Model Results/results.ipynb`

The main results notebook. Loads the dataset, calls `transform_data` and `split` from `eda.py`, trains all six models using `models.py`, prints performance metrics for each, and generates a final comparison chart.

---

## How to Run

> **`eda.py` must be in the same directory as `models.py` and `results.ipynb`** for the imports to work. Copy or move it from the `EDA/` folder before running.

1. Download the dataset: [SeoulBikeData.csv](https://doi.org/10.24432/C5F62R) and place it in the same directory.

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```

3. Run the notebook:
   ```bash
   jupyter notebook "Model Results/results.ipynb"
   ```
   Or open it directly in VS Code / JupyterLab.

---

## Results Summary

| Rank | Model | Test RMSE | Test R² |
|------|-------|-----------|---------|
| 1 | Elastic Net | 0.3676 | 0.8662 |
| 2 | Lasso | 0.3681 | 0.8659 |
| 3 | Ridge | 0.3963 | 0.8445 |
| 4 | SGD | 0.4161 | 0.8286 |
| 5 | KNN | 0.4506 | 0.7990 |
| 6 | OLS | ~1.54 trillion | −2.36×10²⁴ |

All metrics are on the log-transformed target. OLS failed catastrophically on the test set due to near-perfect multicollinearity introduced by the lag features, which it has no mechanism to handle.

---

## Reference

Sathishkumar, V.E. & Cho, Y. (2020). A rule-based model for Seoul bike sharing demand prediction using weather data. *European Journal of Remote Sensing, 53*(sup1), 166–183.
