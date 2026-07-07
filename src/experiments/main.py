"""
pdm run src/experiments/main.py
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
    CSBI_imputation, SFLXRF_imputation, RF_imputation, SFLXDIFF_imputation
)

from src.utils.logger import get_logger
from src.utils.progresser import progress_writer, progress_loader
from src.experiments.experiment_design import create_experiment_design

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

WORKERS = 1

EXPERIMENT_NAME = "test_9"

home = os.getcwd()
experiment_path = os.path.join(home, "export",  EXPERIMENT_NAME)
result_path = os.path.join(experiment_path, "results")

os.makedirs(result_path, exist_ok=True)

progress_csv_path = os.path.join(experiment_path, "progress.csv")

logger = get_logger(log_dir=experiment_path)

os.makedirs(experiment_path, exist_ok=True)

df_exp = create_experiment_design(experiment_path=experiment_path)

df_to_exp = progress_loader(df_experiment_design=df_exp, progress_csv_path=progress_csv_path, logger=logger)

experiment_count = 1


def run_experiment(row_exp):
    try:
        dataset = row_exp["dataset"]
        csv_link = row_exp["csv_link"]
        gap_prc = row_exp["gap_prc"]
        path_to_save = row_exp["path_to_save"]
        col_time = row_exp["col_time"]
        col_target = row_exp["col_target"]

        path_to_save = os.path.join(result_path, path_to_save)
        os.makedirs(path_to_save, exist_ok=True)

        experiment_list = list(range(experiment_count))

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

        plot_distribution_single(
            df_all_values=df_all_values,
            dataset=dataset,
            path_to_save=path_to_save,
            col_target=col_target,
        )

        gap_percent = round(df_all_values[col_target].isna().mean() * 100)

        start_date = df_all_values[col_time].iloc[3]
        start_date = pd.to_datetime(start_date)

        intervals = create_intervals(df_all_values)

        df_AIM_mean_metrics_list = []
        HDIRT_mean_metrics_list = []
        MEAN_mean_metrics_list = []
        MEAN_BETWEEN_mean_metrics_list = []
        SKNN_mean_metrics_list = []
        KNN_mean_metrics_list = []
        LR_mean_metrics_list = []
        LAST_mean_metrics_list = []
        MEDIAN_mean_metrics_list = []
        SMEAN_mean_metrics_list = []
        LINTER_mean_metrics_list = []
        XGB_mean_metrics_list = []
        SFLXGB_mean_metrics_list = []
        POLYNOMIAL_mean_metrics_list = []
        QUADRATIC_mean_metrics_list = []
        CUBIC_mean_metrics_list = []
        SPLINE_mean_metrics_list = []
        LINEAR_mean_metrics_list = []
        CSBI_mean_metrics_list = []
        SFLXRF_mean_metrics_list = []
        RF_mean_metrics_list = []

        SFLXDIFF_mean_metrics_list = []

        for experiment in experiment_list:

            try:

                exp_info = f"Dataset - {dataset} | Percentage gap - {gap_prc} | Experiment № - {experiment}"
                log_info = f"Start: {exp_info}"
                logger.info(log_info)

                random.seed(experiment)
                np.random.seed(experiment)

                df_orig, df_test_with_gaps, drop_indexes = create_error_rate_data(
                    initial_df=df_all_values, test_start_date=start_date,
                     percent_gaps=gap_prc, col_time=col_time, target_value=col_target, intervals=intervals
                )

                plot_dataset(
                    df=df_test_with_gaps,
                    dataset=dataset,
                    path_to_save=path_to_save,
                    gap_prc=gap_prc,
                    experiment=experiment,
                    col_target=col_target,
                    col_time=col_time,
                )

                plot_gap_distribution(
                    df=df_orig[df_orig["is_droped"]],
                    dataset=dataset,
                    path_to_save=path_to_save,
                    gap_prc=gap_prc,
                    experiment=experiment,
                )


                gap_percent = round(df_test_with_gaps[col_target].isna().mean() * 100)


                df_AIM_res = AIM_imputation(df=df_test_with_gaps, col_time=col_time, col_target=col_target)

                df_only_gaps = df_orig.loc[drop_indexes]

                df_orig[col_time] = pd.to_datetime(df_orig[col_time])
                df_test_with_gaps[col_time] = pd.to_datetime(df_test_with_gaps[col_time])

                df_orig = df_orig.set_index(col_time)
                df_test_with_gaps = df_test_with_gaps.set_index(col_time)

                df_only_gaps.set_index(col_time)

                df_AIM = df_AIM_res.copy()
                df_AIM = df_AIM.drop(columns=[col_target,  "interval",  "is_droped"])
                df_AIM_test = pd.concat(
                    [
                        df_orig,
                        df_AIM,
                    ],
                    axis=1
                )
                df_AIM_test = df_AIM_test[df_AIM_test["is_droped"] == True]

                plot_distributions_error(
                    df=df_AIM_test,
                    dataset=dataset,
                    path_to_save=path_to_save,
                    gap_prc=gap_prc,
                    experiment=experiment,
                    col_target=col_target,
                    name="AIM"
                )


                AIM_metrics_df = regression_metrics_by_interval(
                    df=df_AIM_test[df_AIM_test["is_droped"] == True],
                    target_col=col_target,
                    pred_col=f"AIM_{col_target}"
                )

                df_AIM_mean_metrics_list.append(AIM_metrics_df)


                df_HDIRT_res = HDIRT_imputation(df=df_test_with_gaps, col_target=col_target)
                HDIRT_metrics = compute_imputation_metrics(
                    df_imputed=df_HDIRT_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="HDIRT",
                    metrics_func=regression_metrics_by_interval,
                )
                HDIRT_mean_metrics_list.append(HDIRT_metrics)


                df_MEAN_res = MEAN_imputation(df=df_test_with_gaps, col_target=col_target)
                MEAN_metrics = compute_imputation_metrics(
                    df_imputed=df_MEAN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="MEAN",
                    metrics_func=regression_metrics_by_interval,
                )
                MEAN_mean_metrics_list.append(MEAN_metrics)


                df_MEAN_BETWEEN_res = MEAN_BETWEEN_imputation(df=df_test_with_gaps, col_target=col_target)
                MEAN_BETWEEN_metrics = compute_imputation_metrics(
                    df_imputed=df_MEAN_BETWEEN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="MEAN_BETWEEN",
                    metrics_func=regression_metrics_by_interval,
                )
                MEAN_BETWEEN_mean_metrics_list.append(MEAN_BETWEEN_metrics)


                df_SKNN_res = SKNN_imputation(df=df_test_with_gaps, col_target=col_target)
                SKNN_metrics = compute_imputation_metrics(
                    df_imputed=df_SKNN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="SKNN",
                    metrics_func=regression_metrics_by_interval,
                )
                SKNN_mean_metrics_list.append(SKNN_metrics)


                df_KNN_res = KNN_imputation(df=df_test_with_gaps, col_target=col_target)
                KNN_metrics = compute_imputation_metrics(
                    df_imputed=df_KNN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="KNN",
                    metrics_func=regression_metrics_by_interval,
                )
                KNN_mean_metrics_list.append(KNN_metrics)


                df_LR_res = LR_imputation(df=df_test_with_gaps, col_target=col_target)
                LR_metrics = compute_imputation_metrics(
                    df_imputed=df_LR_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="LR",
                    metrics_func=regression_metrics_by_interval,
                )
                LR_mean_metrics_list.append(LR_metrics)


                df_LAST_res = LAST_imputation(df=df_test_with_gaps, col_target=col_target)
                LAST_metrics = compute_imputation_metrics(
                    df_imputed=df_LAST_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="LAST",
                    metrics_func=regression_metrics_by_interval,
                )
                LAST_mean_metrics_list.append(LAST_metrics)


                df_MEDIAN_res = MEDIAN_imputation(df=df_test_with_gaps, col_target=col_target)
                MEDIAN_metrics = compute_imputation_metrics(
                    df_imputed=df_MEDIAN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="MEDIAN",
                    metrics_func=regression_metrics_by_interval,
                )
                MEDIAN_mean_metrics_list.append(MEDIAN_metrics)


                df_SMEAN_res = SMEAN_imputation(df=df_test_with_gaps, col_target=col_target)
                SMEAN_metrics = compute_imputation_metrics(
                    df_imputed=df_SMEAN_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="SMEAN",
                    metrics_func=regression_metrics_by_interval,
                )
                SMEAN_mean_metrics_list.append(SMEAN_metrics)

                df_LINTER_res = LINTER_imputation(df=df_test_with_gaps, col_target=col_target)
                LINTER_metrics = compute_imputation_metrics(
                    df_imputed=df_LINTER_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="LINTER",
                    metrics_func=regression_metrics_by_interval,
                )
                LINTER_mean_metrics_list.append(LINTER_metrics)


                df_XGB_res = XGB_imputation(df=df_test_with_gaps, col_target=col_target)
                XGB_metrics = compute_imputation_metrics(
                    df_imputed=df_XGB_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="XGB",
                    metrics_func=regression_metrics_by_interval,
                )
                XGB_mean_metrics_list.append(XGB_metrics)


                df_POLYNOMIAL_res = POLYNOMIAL_imputation(df=df_test_with_gaps, col_target=col_target)
                POLYNOMIAL_metrics = compute_imputation_metrics(
                    df_imputed=df_POLYNOMIAL_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="POLYNOMIAL",
                    metrics_func=regression_metrics_by_interval,
                )
                POLYNOMIAL_mean_metrics_list.append(POLYNOMIAL_metrics)

                df_QUADRATIC_res = QUADRATIC_imputation(df=df_test_with_gaps, col_target=col_target)
                QUADRATIC_metrics = compute_imputation_metrics(
                    df_imputed=df_QUADRATIC_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="QUADRATIC",
                    metrics_func=regression_metrics_by_interval,
                )
                QUADRATIC_mean_metrics_list.append(QUADRATIC_metrics)


                df_CUBIC_res = CUBIC_imputation(df=df_test_with_gaps, col_target=col_target)
                CUBIC_metrics = compute_imputation_metrics(
                    df_imputed=df_CUBIC_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="CUBIC",
                    metrics_func=regression_metrics_by_interval,
                )
                CUBIC_mean_metrics_list.append(CUBIC_metrics)

                df_SPLINE_res = SPLINE_imputation(df=df_test_with_gaps, col_target=col_target)
                SPLINE_metrics = compute_imputation_metrics(
                    df_imputed=df_SPLINE_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="SPLINE",
                    metrics_func=regression_metrics_by_interval,
                )
                SPLINE_mean_metrics_list.append(SPLINE_metrics)

                df_LINEAR_res = LINEAR_imputation(df=df_test_with_gaps, col_target=col_target)
                LINEAR_metrics = compute_imputation_metrics(
                    df_imputed=df_LINEAR_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="LINEAR",
                    metrics_func=regression_metrics_by_interval,
                )
                LINEAR_mean_metrics_list.append(LINEAR_metrics)

                df_CSBI_res = CSBI_imputation(df=df_test_with_gaps, col_target=col_target)
                CSBI_metrics = compute_imputation_metrics(
                    df_imputed=df_CSBI_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="CSBI",
                    metrics_func=regression_metrics_by_interval,
                )
                CSBI_mean_metrics_list.append(CSBI_metrics)

                df_SFLXRF_res = SFLXRF_imputation(df=df_test_with_gaps, col_target=col_target)
                SFLXRF_metrics = compute_imputation_metrics(
                    df_imputed=df_SFLXRF_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="SFLXRF",
                    metrics_func=regression_metrics_by_interval,
                )
                SFLXRF_mean_metrics_list.append(SFLXRF_metrics)


                df_RF_res = RF_imputation(df=df_test_with_gaps, col_target=col_target)
                RF_metrics = compute_imputation_metrics(
                    df_imputed=df_RF_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="RF",
                    metrics_func=regression_metrics_by_interval,
                )
                RF_mean_metrics_list.append(RF_metrics)


                # df_SFLXDIFF_res = SFLXDIFF_imputation(df=df_test_with_gaps, col_target=col_target)
                # SFLXDIFF_metrics = compute_imputation_metrics(
                #     df_imputed=df_SFLXDIFF_res,
                #     df_orig=df_orig,
                #     col_target=col_target,
                #     col_prefix="SFLXDIFF",
                #     metrics_func=regression_metrics_by_interval,
                # )
                # SFLXDIFF_mean_metrics_list.append(SFLXDIFF_metrics)


                df_SFLXGB_res = SFLXGB_imputation(df=df_test_with_gaps, col_target=col_target)
                SFLXGB_metrics = compute_imputation_metrics(
                    df_imputed=df_SFLXGB_res,
                    df_orig=df_orig,
                    col_target=col_target,
                    col_prefix="SFLXGB",
                    metrics_func=regression_metrics_by_interval,
                )
                SFLXGB_mean_metrics_list.append(SFLXGB_metrics)


                df_SFLXGB = df_SFLXGB_res.copy()
                df_SFLXGB = df_SFLXGB.rename(columns={col_target:f"SFLXGB_{col_target}"})

                df_SFLXGB = df_SFLXGB.drop(columns=["interval",  "is_droped"])
                df_SFLXGB_test = pd.concat(
                    [
                        df_orig,
                        df_SFLXGB,
                    ],
                    axis=1
                )
                df_SFLXGB_test = df_SFLXGB_test[df_SFLXGB_test["is_droped"] == True]

                plot_distributions_error(
                    df=df_SFLXGB_test,
                    dataset=dataset,
                    path_to_save=path_to_save,
                    gap_prc=gap_prc,
                    experiment=experiment,
                    col_target=col_target,
                    name="SFLXGB"
                )
            except Exception as e:
                log_error = f"{exp_info} | ERROR - {e}"
                logger.error(log_error)

            df_all_AIM = pd.concat(df_AIM_mean_metrics_list, ignore_index=True)
            df_all_AIM = df_all_AIM.replace([np.inf, -np.inf], np.nan)
            df_AIM_median = df_all_AIM.groupby("interval", as_index=False).median(numeric_only=True)

            df_all_HDIRT = pd.concat(HDIRT_mean_metrics_list, ignore_index=True)
            df_all_HDIRT = df_all_HDIRT.replace([np.inf, -np.inf], np.nan)
            df_HDIRT_median = df_all_HDIRT.groupby("interval", as_index=False).median(numeric_only=True)

            df_MEAN = pd.concat(MEAN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_MEAN = df_MEAN.groupby("interval", as_index=False).median(numeric_only=True)

            df_MEAN_BETWEEN = pd.concat(MEAN_BETWEEN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_MEAN_BETWEEN = df_MEAN_BETWEEN.groupby("interval", as_index=False).median(numeric_only=True)

            df_SKNN = pd.concat(SKNN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_SKNN = df_SKNN.groupby("interval", as_index=False).median(numeric_only=True)

            df_KNN = pd.concat(KNN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_KNN = df_KNN.groupby("interval", as_index=False).median(numeric_only=True)

            df_LR = pd.concat(LR_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_LR = df_LR.groupby("interval", as_index=False).median(numeric_only=True)

            df_LAST = pd.concat(LAST_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_LAST = df_LAST.groupby("interval", as_index=False).median(numeric_only=True)

            df_MEDIAN = pd.concat(MEDIAN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_MEDIAN = df_MEDIAN.groupby("interval", as_index=False).median(numeric_only=True)

            df_SMEAN = pd.concat(SMEAN_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_SMEAN = df_SMEAN.groupby("interval", as_index=False).median(numeric_only=True)

            df_LINTER = pd.concat(LINTER_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_LINTER = df_LINTER.groupby("interval", as_index=False).median(numeric_only=True)

            df_XGB = pd.concat(XGB_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_XGB = df_XGB.groupby("interval", as_index=False).median(numeric_only=True)

            df_SFLXGB = pd.concat(SFLXGB_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_SFLXGB = df_SFLXGB.groupby("interval", as_index=False).median(numeric_only=True)

            df_POLYNOMIAL = pd.concat(POLYNOMIAL_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_POLYNOMIAL = df_POLYNOMIAL.groupby("interval", as_index=False).median(numeric_only=True)

            df_QUADRATIC = pd.concat(QUADRATIC_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_QUADRATIC = df_QUADRATIC.groupby("interval", as_index=False).median(numeric_only=True)

            df_CUBIC = pd.concat(CUBIC_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_CUBIC = df_CUBIC.groupby("interval", as_index=False).median(numeric_only=True)

            df_SPLINE = pd.concat(SPLINE_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_SPLINE = df_SPLINE.groupby("interval", as_index=False).median(numeric_only=True)

            df_LINEAR = pd.concat(LINEAR_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_LINEAR = df_LINEAR.groupby("interval", as_index=False).median(numeric_only=True)

            df_CSBI = pd.concat(CSBI_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_CSBI = df_CSBI.groupby("interval", as_index=False).median(numeric_only=True)

            df_SFLXRF = pd.concat(SFLXRF_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_SFLXRF = df_SFLXRF.groupby("interval", as_index=False).median(numeric_only=True)

            df_RF = pd.concat(RF_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_RF = df_RF.groupby("interval", as_index=False).median(numeric_only=True)
            #
            # df_SFLXDIFF = pd.concat(SFLXDIFF_mean_metrics_list, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            # df_SFLXDIFF = df_SFLXDIFF.groupby("interval", as_index=False).median(numeric_only=True)

            results = [
                {"method_name": "AIM", "df": df_AIM_median},
                {"method_name": "HDIRT", "df": df_HDIRT_median},
                {"method_name": "MEAN", "df": df_MEAN},
                {"method_name": "MEAN_BETWEEN", "df": df_MEAN_BETWEEN},
                {"method_name": "SKNN", "df": df_SKNN},
                {"method_name": "KNN", "df": df_KNN},
                {"method_name": "LR", "df": df_LR},
                {"method_name": "LAST", "df": df_LAST},
                {"method_name": "MEDIAN", "df": df_MEDIAN},
                {"method_name": "SMEAN", "df": df_SMEAN},
                {"method_name": "LINTER", "df": df_LINTER},
                {"method_name": "XGB", "df": df_XGB},
                {"method_name": "SFLXGB", "df": df_SFLXGB},
                {"method_name": "POLYNOMIAL", "df": df_POLYNOMIAL},
                {"method_name": "QUADRATIC", "df": df_QUADRATIC},
                {"method_name": "CUBIC", "df": df_CUBIC},
                {"method_name": "SPLINE", "df": df_SPLINE},
                {"method_name": "LINEAR", "df": df_LINEAR},
                {"method_name": "CSBI", "df": df_CSBI},
                {"method_name": "SFLXRF", "df": df_SFLXRF},
                {"method_name": "RF", "df": df_RF},
                # {"method_name": "SFLXDIFF", "df": df_SFLXDIFF},
            ]

            os.makedirs(path_to_save, exist_ok=True)

            for r in results:
                file_name = f"{r['method_name']}.csv"
                file_path = os.path.join(path_to_save, file_name)
                r["df"].to_csv(file_path, index=False)

            rows = []

            metrics_cols = ["mae", "rmse", "mape"]

            for r in results:
                df = r["df"].replace([np.inf, -np.inf], np.nan)

                row = {"method": r["method_name"]}

                for m in metrics_cols:
                    row[m] = df[m].mean()

                rows.append(row)


            df_summary = pd.DataFrame(rows)

            df_summary = df_summary.sort_values("mae")

            file_path = os.path.join(path_to_save, "SUMMARY.csv")
            df_summary.to_csv(file_path, index=False)


            plot_metrics_by_interval(
                results=results,
                dataset=dataset,
                path_to_save=path_to_save,
                gap_prc=gap_prc,
            )

        progress_writer(experiment_row=row_exp, experiment_path=experiment_path)
        gc.collect()

    except Exception as e:
        logger.error(e)


for _, row_exp in df_to_exp.iterrows():
    run_experiment(row_exp=row_exp)



# if __name__ == "__main__":
#     rows = df_to_exp.to_dict("records")
#
#     with ProcessPoolExecutor(max_workers=WORKERS) as executor:
#         executor.map(run_experiment, rows)
