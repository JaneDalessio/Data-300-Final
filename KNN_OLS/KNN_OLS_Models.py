# Import all necessary librarys and standard modules
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
    # Print and return RMSE, MAE, R² for train and test sets
    metrics = {}
    # Loops twice: once for train set, once for test set
    # Calculates all of our main metrics for the general evaluation function
    for label, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = np.clip(model.predict(X_), 0, None)
        rmse  = np.sqrt(mean_squared_error(y_, preds))
        mae   = mean_absolute_error(y_, preds)
        r2    = r2_score(y_, preds)
        print(f"  {name} [{label}]  RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
        if label == "Test":
            metrics = {"RMSE": round(rmse, 1), "MAE": round(mae, 1), "R2": round(r2, 3)}
    return metrics


def run_KNN_OLS_models(X_train, X_test, y_train, y_test):
    
    # Trains OLS and KNN models, returns dict of test-set metrics for both models
    
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
        cv = TimeSeriesSplit(n_splits = 3),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    knn_search.fit(X_train, y_train)
    best_k = knn_search.best_params_["knn__n_neighbors"]
    print(f"Best k: {best_k}")
    results["KNN"] = _evaluate("KNN", knn_search.best_estimator_,
                                X_train, y_train, X_test, y_test)

    return results
    

# Main Execution function

if __name__ == "__main__":
    from sklearn.model_selection import TimeSeriesSplit
    import pandas as pd

    df = pd.read_csv("SeoulBikeData.csv", encoding="unicode_escape")
    df = transform_data(df)
    feature_names = [c for c in df.columns if c != "rented_bike_count"]
    tscv = TimeSeriesSplit(n_splits=5)
    X_train, X_test, y_train, y_test = split(df)

    # Run final models and print results
    results = run_KNN_OLS_models(X_train, X_test, y_train, y_test)
    print("\nResults:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")


"""
============================================================
OLS REGRESSION
============================================================
  OLS [Train]  RMSE=0.3  MAE=0.2  R²=0.915
  OLS [Test]  RMSE=571345973489.7  MAE=546516844879.1  R²=-323175673995344056680448.000

============================================================
K-NEAREST NEIGHBORS REGRESSOR
============================================================
Best k: 7
  KNN [Train]  RMSE=0.3  MAE=0.2  R²=0.930
  KNN [Test]  RMSE=0.5  MAE=0.3  R²=0.799

Results:
  OLS: {'RMSE': 571345973489.7, 'MAE': 546516844879.1, 'R2': -3.2317567399534406e+23}
  KNN: {'RMSE': 0.5, 'MAE': 0.3, 'R2': 0.799}
  """
