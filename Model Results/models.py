import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_performance(model_obj, X_train, X_test, Y_train, Y_test,  parameters = None):

    metrics = {}

    for label, X_, Y_ in [("Train", X_train, Y_train),("Test", X_test, Y_test)]:
        preds = np.clip(model_obj.predict(X_), 0, None)
        rmse = np.sqrt(mean_squared_error(Y_, preds))
        mae = mean_absolute_error(Y_, preds)
        r2 = r2_score(Y_, preds)
        metrics[label] = {
            "Parameters": parameters, 
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R\u00b2": round(r2, 4)
        }
    
    return metrics

def clean_print_metrics(metrics):
    print("=" * 30)
    print("Train Metrics")
    print("=" * 30)
    print("Parameters:", metrics['Train']['Parameters'])
    print("RMSE:", metrics['Train']['RMSE'])
    print("MAE:", metrics['Train']['MAE'])
    print("R\u00b2:", metrics['Train']['R\u00b2'])
    print()
    print("=" * 30)
    print("Test Metrics")
    print("=" * 30)
    print("Parameters:", metrics['Test']['Parameters'])
    print("RMSE:", metrics['Test']['RMSE'])
    print("MAE:", metrics['Test']['MAE'])
    print("R\u00b2:", metrics['Test']['R\u00b2'])

def train_ridge(X_train, Y_train):

    tscv = TimeSeriesSplit(n_splits = 3)

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    ridge_params = {"ridge__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]}

    ridge_cv = GridSearchCV(
        ridge_pipeline,
        ridge_params,
        cv = tscv,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    ridge_cv.fit(X_train, Y_train)

    return ridge_cv.best_estimator_, ridge_cv.best_params_

def train_sgd(X_train, Y_train):

    tscv = TimeSeriesSplit(n_splits = 3)

    sgd_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(max_iter = 2000, random_state = 42))
    ])

    sgd_params = {
        "sgd__alpha": [0.0001, 0.001, 0.01, 0.1],
        "sgd__penalty": ["l2", "l1", "elasticnet"]
    }

    sgd_cv = GridSearchCV(
        sgd_pipeline,
        sgd_params,
        cv = tscv,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )
    
    sgd_cv.fit(X_train, Y_train)

    return sgd_cv.best_estimator_, sgd_cv.best_params_

def train_ols(X_train, Y_train):

    ols = Pipeline([
        ("scaler", StandardScaler()),
        ("OLS", LinearRegression())
    ])

    ols.fit(X_train, Y_train)
    
    return ols

def train_knn(X_train, Y_train):

    tscv = TimeSeriesSplit(n_splits = 3)

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_jobs = 1))
    ])

    knn_params = {"knn__n_neighbors": [3, 5, 7]}

    knn_cv = GridSearchCV(
        knn_pipeline,
        knn_params,
        cv = tscv,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    knn_cv.fit(X_train, Y_train)

    return knn_cv.best_estimator_, knn_cv.best_params_

def train_lasso(X_train, Y_train):

    tscv = TimeSeriesSplit(n_splits = 3)

    lasso_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(max_iter = 10000, random_state = 42))
    ])

    lasso_params = {"lasso__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

    lasso_cv = GridSearchCV(
        lasso_pipeline,
        lasso_params,
        cv = tscv,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    lasso_cv.fit(X_train, Y_train)

    return lasso_cv.best_estimator_, lasso_cv.best_params_

def train_elastic_net(X_train, Y_train):

    tscv = TimeSeriesSplit(n_splits = 3)

    elastic_net_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("elastic_net", ElasticNet(max_iter = 10000, random_state = 42))
    ])

    elastic_net_params = {
        "elastic_net__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "elastic_net__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    elastic_net_cv = GridSearchCV(
        elastic_net_pipeline,
        elastic_net_params,
        cv = tscv,
        scoring = "neg_root_mean_squared_error",
        n_jobs = -1
    )

    elastic_net_cv.fit(X_train, Y_train)

    return elastic_net_cv.best_estimator_, elastic_net_cv.best_params_



