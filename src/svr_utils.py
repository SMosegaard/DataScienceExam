import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

all_metrics = []

def svr_forecast_helper(model, last_window, horizon):
    predictions = []
    input_sequence = last_window.copy()
    for _ in range(horizon):
        input_array = np.array(input_sequence[-len(last_window):]).reshape(1, -1)
        next_prediction = model.predict(input_array)[0]
        predictions.append(next_prediction)
        input_sequence.append(next_prediction)
    return predictions


def svr_prep(y_train, window_size):
    X_train, y_target = [], []
    for j in range(len(y_train) - window_size):
        X_train.append(y_train[j:j + window_size])
        y_target.append(y_train[j + window_size])
    return X_train, y_target


def svr_tune(X_train, y_train, dataset, test_size, horizon, window):

    params_of_interest = {"C": [0.1, 1, 10, 100],
                        "epsilon": [0.01, 0.1, 0.5, 1],
                        "gamma": ["scale", "auto", 0.1, 0.5]}
        
    tscv = TimeSeriesSplit(n_splits = 5)
    grid = GridSearchCV(SVR(), params_of_interest, refit = True, cv = tscv, scoring = "neg_mean_squared_error", verbose = 3)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_

    model_params = {"dataset": dataset,
                    "test_size": test_size,
                    "horizon": horizon,
                    "window_size": window,
                    "C": best_params["C"],
                    "epsilon": best_params["epsilon"],
                    "gamma": best_params["gamma"],
                    "best_score_neg_mse": round(grid.best_score_, 3)}
    all_metrics.append(model_params)
    return all_metrics