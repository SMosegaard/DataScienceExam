import pandas as pd
import numpy as np
from ast import literal_eval
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import argparse

all_metrics = []

def parser():
    """
    The user can specify which model to either tune or test when executing specific scripts.
    The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        required = True,
                        choices = ["arima", "lagllama"], # add more
                        help = "Specify which model to tune")
      
    args = parser.parse_args()
    args.model = args.model.lower()
    return args


def data_prep(test, dataset):
    df_train = pd.read_csv(f"../data/train/{dataset}_train.csv")
    df_test = pd.read_csv(f"../data/test/{dataset}_test_{test}.csv")

    if dataset == "weather":
        df_train.rename(columns = {"date": "date", "temperature": "y"}, inplace = True)
        df_test.rename(columns = {"date": "date", "temperature": "y"}, inplace = True)

    if dataset == "carbon":
        df_train.rename(columns = {"date": "date", "carbon_intensity": "y"}, inplace = True)
        df_test.rename(columns = {"date": "date", "carbon_intensity": "y"}, inplace = True)

    return df_train, df_test


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


def convert_tuple(a):
    """ Convert input to tuple """
    return literal_eval(a)


def extract_hyperparams_arima(hyperparams, horizon):
    """ Extract the hyperparameter order and seasonal_order from the df and given horizon """
    mask = hyperparams["horizon"] == horizon
    order = hyperparams.loc[mask, "order"]
    seasonal_order = hyperparams.loc[mask, "seasonal_order"]
    order, seasonal_order = order.values[0], seasonal_order.values[0]
    return order, seasonal_order


def rolling_origin_eval_prep(df_train, df_test, horizon):
    train_split, test_split = [], [] # initialize empty lists to store each train and test split
    for i in range(0, len(df_test) - horizon + 1):
        curr_train = pd.concat([df_train, df_test.iloc[:i]], axis = 0)
        curr_test = df_test.iloc[i:i + horizon] # create a test windos from i to i+horizon, meaning select the next "horizon step" of test data starting from i
        train_split.append(curr_train)
        test_split.append(curr_test)
    return train_split, test_split


def cal_smape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast) # convert actual and forecasted values to numpy arrays
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    difference = np.abs(actual - forecast) / denominator
    difference = np.where(denominator == 0, 0, difference) # to avoid dividing by 0 or NaN
    return 100 * np.mean(difference)


def evaluate(actual, forecast):
    mae = round(mean_absolute_error(actual, forecast), 3)
    mse = round(mean_squared_error(actual, forecast), 3)
    rmse = round(np.sqrt(mse), 3)
    smape = round(cal_smape(actual, forecast), 3)
    return mae, mse, rmse, smape