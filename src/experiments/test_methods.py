"""
pdm run src/experiments/test_methods.py
"""
import os
import gc
import random
import numpy as np
import pandas as pd

# from concurrent.futures import ProcessPoolExecutor

from src.processing.data_processing import (
    create_error_rate_data, create_intervals, regression_metrics_by_interval,
    plot_dataset, plot_gap_distribution, plot_distributions_error, plot_distribution_single,
    plot_metrics_by_interval, compute_imputation_metrics
)
from src.methods.imputation_methods import (
    AIM_imputation, HDIRT_imputation, LR_imputation, SKNN_imputation,
    KNN_imputation, MEAN_BETWEEN_imputation, MEAN_imputation,
    LAST_imputation, MEDIAN_imputation, SMEAN_imputation,
    LINTER_imputation, XGB_imputation, SFLXGB_imputation, POLYNOMIAL_imputation,
    QUADRATIC_imputation, CUBIC_imputation, SPLINE_imputation, LINEAR_imputation,
    CSBI_imputation, SFLXRF_imputation, RF_imputation, SFLXDIFF_imputation, PFBGB_imputation, PFBRF_imputation
)


def run_experiment(row_exp):
    try:
        csv_link = row_exp["csv_link"]
        gap_prc = row_exp["gap_prc"]
        col_time = row_exp["col_time"]
        col_target = row_exp["col_target"]

        df_all_values = pd.read_csv(csv_link)
        df_all_values[col_time] = pd.to_datetime(df_all_values[col_time])

        df_all_values = df_all_values[[col_time, col_target]]
        df_all_values = df_all_values.drop_duplicates(subset=[col_time], keep="first")


        df_all_values[col_target] = (
                                            df_all_values[col_target] - df_all_values[col_target].min()
                                    ) / (
                                            df_all_values[col_target].max() - df_all_values[col_target].min()
                                    )

        df_all_values = df_all_values.sort_values(col_time)
        df_all_values = df_all_values.dropna(subset=[col_target])

        experiment = 1

        random.seed(experiment)
        np.random.seed(experiment)

        start_date = df_all_values[col_time].iloc[3]
        start_date = pd.to_datetime(start_date)

        intervals = create_intervals(df_all_values)

        df_orig, df_test_with_gaps, drop_indexes = create_error_rate_data(
            initial_df=df_all_values, test_start_date=start_date,
            percent_gaps=gap_prc, col_time=col_time, target_value=col_target, intervals=intervals
        )

        df_orig[col_time] = pd.to_datetime(df_orig[col_time])
        df_test_with_gaps[col_time] = pd.to_datetime(df_test_with_gaps[col_time])

        df_orig = df_orig.set_index(col_time)
        df_test_with_gaps = df_test_with_gaps.set_index(col_time)


        """ ========================================= METHOD ========================================="""

        # df_METHOD_res = PFBGB_imputation(df=df_test_with_gaps, col_target=col_target)
        # df_METHOD_res = SKNN_imputation(df=df_test_with_gaps, col_target=col_target)
        df_METHOD_res = PFBRF_imputation(df=df_test_with_gaps, col_target=col_target)


        # df_METHOD_res = AIM_imputation(df=df_test_with_gaps, col_time=col_time, col_target=col_target,)

        """ =========================================================================================="""
        print(df_METHOD_res[df_METHOD_res["is_droped"]])

        # df_METHOD_res[col_time] = df_METHOD_res.index
        # df_METHOD_res = df_METHOD_res.reset_index(drop=True)
        # df_METHOD_res = df_METHOD_res.set_index(col_time)


        df_METHOD_res = df_METHOD_res.rename(columns={col_target: f"pred_{col_target}"})


        df_METHOD = df_METHOD_res.copy()

        df_METHOD = df_METHOD.drop(columns=["interval",  "is_droped"])

        df_METHOD_test = pd.concat(
            [
                df_orig,
                df_METHOD,
            ],
            axis=1
        )

        df_METHOD_test = df_METHOD_test[df_METHOD_test["is_droped"] == True]


        METHOD_metrics_df = regression_metrics_by_interval(
            df=df_METHOD_test[df_METHOD_test["is_droped"] == True],
            target_col=col_target,
            pred_col=f"pred_{col_target}"
        )

        return METHOD_metrics_df

    except Exception as e:
        print(e)

row_exp = {
    "dataset": "russia_amur_region",
    "gap_prc": 10,
    "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnke5YReb1RRwN6ZynJSxoUcXnr2buEXP45mOtnv7X3neOQhT-_vMPnIB4lUBpeu0nQaqWq6awDvtG/pub?gid=935054795&single=true&output=csv",
    "col_time": "datetime",
    "col_target": "VC_факт",
}


METHOD_metrics_df = run_experiment(row_exp=row_exp)

print(METHOD_metrics_df)
mean_mape = METHOD_metrics_df["mape"].mean()
mean_rmse = METHOD_metrics_df["rmse"].mean()
mean_mae = METHOD_metrics_df["mae"].mean()

print(f"MAPE = {mean_mape} | RMSE = {mean_rmse} | MAE = {mean_mae}")


