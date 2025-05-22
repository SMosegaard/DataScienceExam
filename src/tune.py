import os
import pandas as pd
import torch
from gluonts.evaluation import Evaluator

from arima_utils import data_prep, eval_arima_parms, parser
from lagllama_utils import prep_gluonts_df, lagllama_estimator 

os.chdir("../..") # navigating out of the lag-llama repo. Current wd is now 'DataScienceExam/src/' 

dataset_label = ["weather", "carbon"]
test_size_label = ["small", "large"]
horizons = [5, 50]

all_metrics = []

# for the Lag-Llama model: listed hyperparams of interest
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

            df_train, df_test = data_prep(test_size, dataset)

            if args.model == "arima":
                                    
                all_metrics = eval_arima_parms(dataset, df_train, horizon, test_size)

            if args.model == "lagllama":
                                    
                gluonts_df = prep_gluonts_df(df_train, freq = "H")

                for rope in rope_scaling:
                    for c_len in context_length:

                        print(f"Context length {c_len} and rope scaling {rope}\n")                        
                        predicted_series, actual_series = lagllama_estimator(gluonts_df, horizon, torch.device('cpu'), c_len, rope)

                        evaluator = Evaluator()
                        metrics, _ = evaluator(iter(actual_series), iter(predicted_series))

                        params_dict = {"dataset": dataset,
                                        "test_size": test_size,
                                        "horizon": horizon,
                                        "context_length": c_len,
                                        "rope_scaling": rope}
                        
                        metrics.update(params_dict)
                        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    if args.model == "arima":
        metrics_df.to_csv("../out/arima_hyperparameters.csv", index = False)
    if args.model == "lagllama":
        best_params = metrics_df.loc[metrics_df.groupby(["dataset", "test_size", "horizon"])["MSE"].idxmin()]
        best_params.to_csv("../out/lagllama_hyperparameters.csv", index = False)

    print("The tuning has been conducted and the results have been saved in the out folder")


if __name__ == "__main__":
    main()