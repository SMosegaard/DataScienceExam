import os
import pandas as pd
import torch

from utils import parser, data_prep
from arima_utils import eval_arima_parms
from svr_utils import svr_prep, svr_tune
from lagllama_utils import prep_gluonts_df, lagllama_estimator, eval_lagllama

os.chdir("../..") # navigating out of the lag-llama repo. Current wd is now 'DataScienceExam/src/' 

dataset_label = ["weather", "carbon"]
test_size_label = ["small", "large"]
horizons = [5, 50]

# for the SVR and Lag-Llama model: listed hyperparams of interest
window_size = [3, 5, 7, 10, 24]
rope_scaling = [True, False]
context_length = [32, 64, 128, 256, 512] # suggested values: https://github.com/time-series-foundation-models/lag-llama

def main():

    args = parser()

    for dataset in dataset_label:
        for horizon in horizons:
            if horizon == 5:
                test_size = "small"
            elif horizon == 50:
                test_size = "large"
            
            print(f"Conducting hyperparameter tuning on model {args.model}\n")
            print(f"{dataset} dataset: running horizon {horizon} (test size {test_size})\n")

            df_train, _ = data_prep(test_size, dataset)

            if args.model == "arima":
                                    
                all_metrics = eval_arima_parms(dataset, df_train, horizon, test_size)

            elif args.model == "svr":

                y_train = df_train["y"].values
                for window in window_size:
                    X_train, y_target = svr_prep(y_train, window)
                    all_metrics = svr_tune(X_train, y_target, dataset, test_size, horizon, window)

            elif args.model == "lagllama":
                                    
                gluonts_df = prep_gluonts_df(df_train, freq = "H")
                for rope in rope_scaling:
                    for c_len in context_length:
                        predicted_series, actual_series = lagllama_estimator(gluonts_df, horizon, torch.device('cpu'), c_len, rope)
                        all_metrics = eval_lagllama(predicted_series, actual_series, dataset, test_size, horizon, c_len, rope)

    metrics_df = pd.DataFrame(all_metrics)

    if args.model == "arima":
        metrics_df.to_csv("../out/arima_hyperparameters.csv", index = False)

    elif args.model == "svr":
        metrics_df["mse"] = -metrics_df["best_score_neg_mse"]
        best_params_df = metrics_df.loc[metrics_df.groupby(["dataset", "horizon"])["mse"].idxmin()].reset_index(drop = True)
        best_params_df.to_csv("../out/svr_hyperparameters.csv", index = False)
    
    elif args.model == "lagllama":
        best_params = metrics_df.loc[metrics_df.groupby(["dataset", "test_size", "horizon"])["MSE"].idxmin()]
        best_params.to_csv("../out/lagllama_hyperparameters.csv", index = False)

    print("The tuning has been conducted and the results have been saved in the out folder")


if __name__ == "__main__":
    main()