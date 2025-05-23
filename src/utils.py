import pandas as pd
import numpy as np
from ast import literal_eval
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse


def parser():
    """
    The user can specify which model to either tune or test when executing specific scripts.
    The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        required = True,
                        choices = ["arima", "svr", "lagllama"], # add more
                        help = "Specify which model to tune or test")
      
    args = parser.parse_args()
    args.model = args.model.lower()
    return args


def data_prep(test, dataset):
    """
    The function prepares the data. Specifically, it loads the train and test data from the data
    folder and renames the columns of interest depending on the current dataset.
    """
    df_train = pd.read_csv(f"../data/train/{dataset}_train.csv")
    df_test = pd.read_csv(f"../data/test/{dataset}_test_{test}.csv")

    if dataset == "weather":
        df_train.rename(columns = {"date": "date", "temperature": "y"}, inplace = True)
        df_test.rename(columns = {"date": "date", "temperature": "y"}, inplace = True)

    if dataset == "carbon":
        df_train.rename(columns = {"date": "date", "carbon_intensity": "y"}, inplace = True)
        df_test.rename(columns = {"date": "date", "carbon_intensity": "y"}, inplace = True)

    return df_train, df_test


def convert_tuple(a):
    """ Convert input to tuple """
    return literal_eval(a)


def rolling_origin_eval_prep(df_train, df_test, horizon):
    """
    The function prepares the rolling-origin evaluation strategy. It starts by
    initializing empty lists to store each train and test split. Then, the function
    iteratievly expands the training data by including more of the test data, and
    creates a corresponding test window from i+ horizon. This means, that
    it selects the new "horizon step" of the test data starting from i.
    The function returns lists of train and test splits for each evaluation step. 
    """
    train_split, test_split = [], []
    for i in range(0, len(df_test) - horizon + 1):
        curr_train = pd.concat([df_train, df_test.iloc[:i]], axis = 0)
        curr_test = df_test.iloc[i:i + horizon]
        train_split.append(curr_train)
        test_split.append(curr_test)
    return train_split, test_split


def cal_smape(actual, forecast):
    """
    This function calculates the sMAPE between the actual and forecasted values.
    The function converts the actual and forecasted values to numpy arrays, computes the absolute
    error for each pair, and returns the average error as a percentage. 
    """
    actual, forecast = np.array(actual), np.array(forecast)
    denominator = (np.abs(forecast) + np.abs(actual)) / 2
    difference = np.abs(forecast - actual) / denominator
    return 100 * np.mean(difference)


def evaluate(actual, forecast):
    """
    This function calculates MAE, MSE, RMSE, and sMAPE based
    on actual and forecasted values
    """
    mae = round(mean_absolute_error(actual, forecast), 3)
    mse = round(mean_squared_error(actual, forecast), 3)
    rmse = round(np.sqrt(mse), 3)
    smape = round(cal_smape(actual, forecast), 3)
    return mae, mse, rmse, smape