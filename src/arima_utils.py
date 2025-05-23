import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

all_metrics = []

def eval_arima_parms(data, df_train, horizon, test):
    model = auto_arima(df_train["y"],
                       start_p = 1,
                       start_q = 1,
                       start_P = 0,
                       max_p = 10,
                       max_q = 10,
                       m = 24,
                       information_criterion = "oob",
                       stepwise = False,
                       error_action = "ignore",
                       trace = True,
                       out_of_sample_size = horizon)

    model_params = model.get_params()
    
    for var in ["maxiter", "method", "out_of_sample_size", "scoring", "scoring_args", "start_params", "suppress_warnings"]:
        model_params.pop(var)

    model_dict = model.to_dict()
    model_params["mse"] = model_dict["oob"]

    params_dict = {"dataset": data,
                   "test_size": test,
                   "horizon": horizon}

    model_params.update(params_dict)
    all_metrics.append(model_params)
    
    return all_metrics


def test_arima(curr_train, curr_test, order, seasonal_order, horizon):
    model = ARIMA(curr_train["y"], order = order, seasonal_order = seasonal_order).fit()
    forecast = model.forecast(steps = horizon)
    actual = curr_test["y"].values
    return forecast, actual


def extract_hyperparams_arima(hyperparams, horizon):
    """ Extract the hyperparameter order and seasonal_order from the df and given horizon """
    mask = hyperparams["horizon"] == horizon
    order = hyperparams.loc[mask, "order"]
    seasonal_order = hyperparams.loc[mask, "seasonal_order"]
    order, seasonal_order = order.values[0], seasonal_order.values[0]
    return order, seasonal_order
