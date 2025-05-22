import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

model_list = ["arima", "lagllama", "svr"]
metrics = ["mae", "rmse", "smape"]
group = ["dataset", "horizon"]

dataset_label = ["weather", "carbon"]
test_size_label = ["small", "large"]
horizons = [5, 50]


def main():

    for model in model_list:
        results = pd.read_csv(f"../out/{model}_results.csv")
        mean_metrics_df = results.groupby(group)[metrics].agg(["mean", "std"])
        mean_metrics_df = mean_metrics_df.reset_index()
        mean_metrics_df_rounded = round(mean_metrics_df, 3)
        mean_metrics_df_rounded.to_csv(f"../out/{model}_results_agg.csv")


    for dataset in dataset_label:
        for horizon in horizons:
            if horizon == 5:
                test_size = "small"
            elif horizon == 50:
                test_size = "large"
            
            # Actual values:
            train = pd.read_csv(f"../data/train/{dataset}_train.csv")
            test = pd.read_csv(f"../data/test/{dataset}_test_{test_size}.csv")

           # Forecasted values:
            forecast_arima = pd.read_csv(f"../out/arima_{horizon}_{dataset}.csv")
            forecast_svr = pd.read_csv(f"../out/svr_{horizon}_{dataset}.csv")
            forecast_lagllama = pd.read_csv(f"../out/lagllama_{horizon}_{dataset}.csv")

            if dataset == "weather":
                actual_y_train, actual_y_test = train["temperature"], test["temperature"].iloc[:horizon]
                ylabel = "Temperature"
                title = "Temperature Forecast"

            if dataset == "carbon":
                actual_y_train, actual_y_test = train["carbon_intensity"], test["carbon_intensity"].iloc[:horizon]
                ylabel = "Carbon Intensity"
                title = "Carbon Intensity Forecast"

            actual_y = pd.concat([actual_y_train, actual_y_test]).reset_index(drop = True)
            dates = pd.concat([train["date"], test["date"].iloc[:horizon]]).reset_index(drop = True)

            forecast_arima = forecast_arima["0"].iloc[:horizon].reset_index(drop = True)
            forecast_svr = forecast_svr["0"].iloc[:horizon].reset_index(drop = True)
            forecast_lagllama = forecast_lagllama["0"].iloc[:horizon].reset_index(drop = True)
            forecast_dates = test["date"].iloc[:horizon].reset_index(drop = True)

            # Plot
            fig, ax = plt.subplots(figsize = (14, 6))
            ax.plot(dates, actual_y, label = "True", color = "seagreen")
            ax.plot(forecast_dates, forecast_arima, label = "ARIMA", color = "palevioletred")
            ax.plot(forecast_dates, forecast_svr, label = "SVR", color = "goldenrod")
            ax.plot(forecast_dates, forecast_lagllama, label = "LagLlama", color = "steelblue")

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

            plt.xlabel("Time", fontsize  = 12)
            plt.ylabel(ylabel, fontsize  = 12)
            plt.title(title, fontsize = 14)

            if dataset == "weather":
                plt.ylim(-10, 20)
            if dataset == "carbon":
                plt.ylim(0, 160)
            
            plt.xticks(rotation = 45)
            plt.tight_layout()
            plt.legend(loc = "upper right")
            plt.savefig(f"../plots/actual_vs_forecast_{dataset}_{horizon}.png")
            plt.show()

if __name__ == "__main__":
    main()