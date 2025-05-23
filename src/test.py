import os
import pandas as pd
import numpy as np
from gluonts.evaluation import Evaluator
from sklearn.svm import SVR

from utils import parser, data_prep, rolling_origin_eval_prep, convert_tuple, evaluate
from arima_utils import extract_hyperparams_arima, test_arima
from svr_utils import svr_prep, extract_hyperparams_svr, svr_forecast_helper
from lagllama_utils import extract_hyperparams_lagllama, prep_gluonts_df, lagllama_estimator

os.chdir("../..") # navigating out of the lag-llama repo. Current wd is now 'DataScienceExam/src/' 


dataset_label = ["weather", "carbon"]
test_size_label = ["small", "large"]
horizons = [5, 50]
all_metrics = []


def main():

    args = parser()
    
    hyperparams = pd.read_csv(f"../out/{args.model}_hyperparameters.csv")

    for dataset in dataset_label:
        for horizon in horizons:
            if horizon == 5:
                test_size = "small"
            elif horizon == 50:
                test_size = "large"
            
            print(f"Testing {args.model}!\n")
            print(f"{dataset} dataset: running horizon {horizon} (test size {test_size})\n")

            df_train, df_test = data_prep(test_size, dataset)

            if args.model == "arima":

                order, seasonal_order = extract_hyperparams_arima(hyperparams, horizon)
                order, seasonal_order = convert_tuple(order), convert_tuple(seasonal_order)
                train_split, test_split = rolling_origin_eval_prep(df_train, df_test, horizon)
        
                forecast_df = pd.DataFrame()
                forecast_df["date"] = df_test.index
                forecast_df["y"] = df_test["y"].values

                for i, (curr_train, curr_test) in enumerate(zip(train_split, test_split)):
                    
                    curr_train.reset_index(drop = True)
                    forecast, actual = test_arima(curr_train, curr_test, order, seasonal_order, horizon)

                    i_forecast = np.concatenate([np.repeat(np.nan, i), forecast, np.repeat(np.nan, len(df_test) - i - horizon)])
                    forecast_df[f"{i}"] = i_forecast

                    mae, mse, rmse, smape = evaluate(actual, forecast)

                    metadata_dict = {"dataset": dataset,
                                    "test_size": test_size,
                                    "horizon": horizon,
                                    "order": order,
                                    "seasonal_order": seasonal_order,
                                    "iter": i,
                                    "mae": mae,
                                    "mse": mse,
                                    "rmse": rmse,
                                    "smape": smape}

                    forecast_df.to_csv(f"../out/{args.model}_{horizon}_{dataset}.csv", index = False)
                    all_metrics.append(metadata_dict)

            elif args.model == "svr":
                window_size, C_param, epsilon, gamma = extract_hyperparams_svr(hyperparams, horizon)
                train_split, test_split = rolling_origin_eval_prep(df_train, df_test, horizon)

                forecast_df = pd.DataFrame()
                forecast_df["date"] = df_test.index
                forecast_df["y"] = df_test["y"].values

                for i, (curr_train, curr_test) in enumerate(zip(train_split, test_split)):

                    y_train = curr_train["y"].values
                    X_train, y_target = svr_prep(y_train, window_size)
                    
                    model = SVR(gamma = gamma, C = C_param, epsilon = epsilon)
                    model.fit(X_train, y_target)

                    last_window = list(y_train[-window_size:])
                    forecast = svr_forecast_helper(model, last_window, horizon)
                    actual = curr_test["y"].values
                    i_forecast = np.concatenate([np.repeat(np.nan, i), forecast, np.repeat(np.nan, len(df_test) - i - horizon)])
                    forecast_df[f"{i}"] = i_forecast

                    mae, mse, rmse, smape = evaluate(actual, forecast)

                    metadata_dict = {"dataset": dataset,
                                    "test_size": test_size,
                                    "horizon": horizon,
                                    "window_size": window_size,
                                    "C": C_param,
                                    "epsilon": epsilon,
                                    "gamma": gamma,
                                    "iter": i,
                                    "mae": round(mae),
                                    "mse": round(mse),
                                    "rmse": round(rmse),
                                    "smape": round(smape)}
                    
                    forecast_df.to_csv(f"../out/{args.model}_{horizon}_{dataset}.csv", index = False)
                    all_metrics.append(metadata_dict)
                
            elif args.model == "lagllama":
                
                rope, c_len = extract_hyperparams_lagllama(hyperparams, horizon)
                train_split, test_split = rolling_origin_eval_prep(df_train, df_test, horizon)

                forecast_df = pd.DataFrame()
                forecast_df["date"] = df_test.index
                forecast_df["y"] = df_test["y"].values

                for i, (curr_train, curr_test) in enumerate(zip(train_split, test_split)):
                    
                    gluonts_df = prep_gluonts_df(curr_train, freq = "H")
                    predicted_series, actual_series = lagllama_estimator(gluonts_df, horizon, torch.device('cpu'), c_len, rope)
                    
                    evaluator = Evaluator()
                    metrics, _ = evaluator(iter(actual_series), iter(predicted_series))
                    
                    prediction_m = predicted_series[0].mean
                    i_forecast = np.concatenate([np.repeat(np.nan, i), prediction_m, np.repeat(np.nan, len(df_test) - i - horizon)])
                    forecast_df[f"{i}"] = i_forecast

                    metadata_dict = {"dataset": dataset,
                                    "test_size": test_size,
                                    "horizon": horizon,
                                    "context_length": c_len,
                                    "rope_scaling": rope,
                                    "iter": i,
                                    "mae": round(metrics["abs_error"], 3),
                                    "mse": round(metrics["MSE"], 3),
                                    "rmse": round(metrics["RMSE"], 3),
                                    "smape": round(metrics["sMAPE"], 3)}

                    forecast_df.to_csv(f"../out/{args.model}_{horizon}_{dataset}.csv", index = False)
                    all_metrics.append(metadata_dict)


    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"../out/{args.model}_results.csv")


if __name__ == "__main__":
    main()