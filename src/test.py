import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from gluonts.evaluation import Evaluator

from arima_utils import parser, data_prep, extract_hyperparams_arima, rolling_origin_eval_prep, evaluate, convert_tuple
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
                    model = ARIMA(curr_train["y"], order = order, seasonal_order = seasonal_order).fit()

                    forecast = model.forecast(steps = horizon)
                    actual = curr_test["y"].values

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


            if args.model == "lagllama":
                
                rope, c_len = extract_hyperparams_lagllama(hyperparams, horizon)
                rope, c_len = convert_tuple(rope), convert_tuple(c_len)
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
                    
                    mae = metrics["abs_error"] * (metrics["abs_target_mean"] / metrics["abs_target_sum"])

                    metadata_dict = {"dataset": dataset,
                                    "test_size": test_size,
                                    "horizon": horizon,
                                    "context_length": c_len,
                                    "rope_scaling": rope,
                                    "iter": i,
                                    "mae": round(mae, 3),
                                    "mse": round(metrics["MSE"], 3),
                                    "rmse": round(metrics["RMSE"], 3),
                                    "smape": round(metrics["sMAPE"], 3)}

                    forecast_df.to_csv(f"../out/{args.model}_{horizon}_{dataset}.csv", index = False)
                    all_metrics.append(metadata_dict)


    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"../out/{args.model}_results.csv")



if __name__ == "__main__":
    main()