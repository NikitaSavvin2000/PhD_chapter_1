import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.processing.data_processing import (
    create_error_rate_data, create_intervals,
    calculate_mape_improved, plot_interval_distribution,
    AIM_create_test_nan, add_additional_features,
    build_gap_pairs, merge_singletons, get_gap_classes,
    build_ranges, mae, rmse, mape, assign_gap_classes,
    calc_metrics_by_class, get_gap_intervals, get_gap_class,
    apply_best_method_by_class, build_training_data, get_lag_vector,
    hdirt_fallback
)


DEFAULT_XGB_PARAMS = {
    "max_depth": 5,
    "n_estimators": 500,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "min_child_weight": 7,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "booster": "gbtree",
    "objective": "reg:squarederror"
}


def SFLXGB_imputation(
        df,
        col_target,
        lag=100,
        window_years=3,
        params=None
):
    params = params or DEFAULT_XGB_PARAMS

    df = df.copy().sort_index()
    values = df[col_target].astype(float).values

    X_train, y_train = build_training_data(values, lag)

    if len(X_train) < 10:
        return df

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    for i in range(len(values)):

        if not np.isnan(values[i]):
            continue

        lag_vec = get_lag_vector(values, i, lag)

        if lag_vec is not None:
            pred = model.predict(lag_vec.reshape(1, -1))[0]
            values[i] = pred
            continue

        if i - lag < 0:
            fallback = hdirt_fallback(values, i, window_years)

            if fallback is not None:
                values[i] = fallback

    df[col_target] = values

    return df


def XGB_imputation(
        df,
        col_target,
        params=None
):
    params = params or DEFAULT_XGB_PARAMS

    df = df.copy().sort_index()
    values = df[col_target].astype(float).values

    X_train, y_train = [], []

    for i in range(1, len(values)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue

        X_train.append([values[i - 1]])
        y_train.append(values[i])

    if len(X_train) < 10:
        return df

    model = XGBRegressor(**params)
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32))

    for i in range(1, len(values)):
        if not np.isnan(values[i]):
            continue

        if not np.isnan(values[i - 1]):
            values[i] = model.predict([[values[i - 1]]])[0]

    df[col_target] = values

    return df

def HDIRT_imputation(df, col_target, time_window_years=3):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    for time_point in tqdm(filled_df[filled_df[col_target].isna()].index):

        data = []
        target = []

        for i in range(1, time_window_years + 1):

            previous_year = time_point - pd.DateOffset(years=i)
            next_year = time_point + pd.DateOffset(years=i)

            if previous_year in index:
                val = filled_df.loc[previous_year, col_target]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if not pd.isna(val):
                    data.append([-i])
                    target.append(val)
                    break

        for i in range(1, time_window_years + 1):

            next_year = time_point + pd.DateOffset(years=i)

            if next_year in index:
                val = filled_df.loc[next_year, col_target]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if not pd.isna(val):
                    data.append([i])
                    target.append(val)
                    break

        if len(data) >= 2:
            regressor = LinearRegression()
            regressor.fit(np.array(data), target)
            filled_df.at[time_point, col_target] = regressor.predict(np.array([[0]]))[0]

        else:
            interp = filled_df[col_target].interpolate(method="linear")
            val = interp.loc[time_point]

            if isinstance(val, pd.Series):
                val = val.iloc[0]

            filled_df.at[time_point, col_target] = val

    return filled_df


def LR_imputation(df, col_target):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    known = filled_df[filled_df[col_target].notna()]
    missing = filled_df[filled_df[col_target].isna()]

    if len(known) == 0:
        return filled_df

    X_train = np.arange(len(known)).reshape(-1, 1)
    y_train = known[col_target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = np.arange(len(missing)).reshape(-1, 1)
    preds = model.predict(X_missing)

    for i, idx in enumerate(missing.index):
        filled_df.at[idx, col_target] = preds[i]

    return filled_df


def SKNN_imputation(df, col_target='P_l', k=3, time_window_years=3):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    missing_data = []
    missing_indices = []

    for time_point in tqdm(filled_df[filled_df[col_target].isna()].index, desc="Preparing data"):

        row = []

        for i in range(1, time_window_years + 1):

            prev_year = time_point - pd.DateOffset(years=i)

            if prev_year in index:
                val = filled_df.loc[prev_year, col_target]

                if isinstance(val, pd.Series):
                    val = val.iloc[0]

                row.append(val)
            else:
                row.append(np.nan)

        row.append(np.nan)

        missing_data.append(row)
        missing_indices.append(time_point)

    if not missing_data:
        return filled_df

    imputer = KNNImputer(n_neighbors=k)

    filled_data = imputer.fit_transform(missing_data)

    for i, idx in tqdm(enumerate(missing_indices), desc="Applying KNN imputation"):

        filled_df.at[idx, col_target] = filled_data[i, -1]

    return filled_df


def KNN_imputation(df, col_target='P_l', k=3, time_window_years=3):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    missing_data = []
    missing_indices = []

    for time_point in tqdm(filled_df[filled_df[col_target].isna()].index, desc="Preparing data"):

        row = []

        # берем лаги по времени как признаки (как в SKNN)
        for i in range(1, time_window_years + 1):

            prev_time = time_point - pd.DateOffset(years=i)

            if prev_time in index:
                val = filled_df.loc[prev_time, col_target]

                if isinstance(val, pd.Series):
                    val = val.iloc[0]

                row.append(val)
            else:
                row.append(np.nan)

        # добавляем таргет как последний столбец
        row.append(np.nan)

        missing_data.append(row)
        missing_indices.append(time_point)

    if not missing_data:
        return filled_df

    imputer = KNNImputer(n_neighbors=k)

    filled_data = imputer.fit_transform(missing_data)

    for i, idx in tqdm(enumerate(missing_indices), desc="Applying KNN imputation"):

        filled_df.at[idx, col_target] = filled_data[i, -1]

    return filled_df


def MEAN_BETWEEN_imputation(df, col_target):
    filled_df = df.copy()

    series = filled_df[col_target]

    for idx in filled_df[series.isna()].index:

        values = filled_df[col_target]

        left = values.loc[:idx].last_valid_index()
        right = values.loc[idx:].first_valid_index()

        if left is not None and right is not None:
            left_val = values.loc[left]
            right_val = values.loc[right]

            if hasattr(left_val, "iloc"):
                left_val = left_val.iloc[0]
            if hasattr(right_val, "iloc"):
                right_val = right_val.iloc[0]

            value = (left_val + right_val) / 2

        elif left is not None:
            value = values.loc[left]
            if hasattr(value, "iloc"):
                value = value.iloc[0]

        elif right is not None:
            value = values.loc[right]
            if hasattr(value, "iloc"):
                value = value.iloc[0]

        else:
            continue

        filled_df.at[idx, col_target] = float(value)

    return filled_df


def MEAN_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].fillna(filled_df[col_target].mean())
    return filled_df


def LAST_imputation(df, col_target):

    df_init_filled_LR = df.copy()
    df_init_filled_LR = df_init_filled_LR.sort_index()

    df_init_filled_LR[col_target] = df_init_filled_LR[col_target].ffill()

    return df_init_filled_LR


def MEDIAN_imputation(df, col_target):

    df_init_filled_MED = df.copy()
    df_init_filled_MED = df_init_filled_MED.sort_index()

    median_val = df_init_filled_MED[col_target].median()
    df_init_filled_MED[col_target] = df_init_filled_MED[col_target].fillna(median_val)

    return df_init_filled_MED


def SMEAN_imputation(df, col_target):

    df_init_filled_SEA = df.copy()
    df_init_filled_SEA = df_init_filled_SEA.sort_index()

    if not isinstance(df_init_filled_SEA.index, pd.DatetimeIndex):
        df_init_filled_SEA.index = pd.to_datetime(df_init_filled_SEA.index)

    season = df_init_filled_SEA.index.month

    seasonal_means = df_init_filled_SEA.groupby(season)[col_target].transform("mean")

    df_init_filled_SEA[col_target] = df_init_filled_SEA[col_target].fillna(seasonal_means)

    return df_init_filled_SEA


def LINTER_imputation(df, col_target, window=5):

    df_init_filled_LOC = df.copy()
    df_init_filled_LOC = df_init_filled_LOC.sort_index()

    local_mean = df_init_filled_LOC[col_target].rolling(
        window=window,
        min_periods=1,
        center=True
    ).mean()

    filled = df_init_filled_LOC[col_target].fillna(local_mean)

    filled = filled.fillna(method="ffill").fillna(method="bfill")

    df_init_filled_LOC[col_target] = filled

    return df_init_filled_LOC


def AIM_imputation(df, col_time, col_target):

    gap_classes = get_gap_classes(df=df, col_target=col_target)

    print(f"gap_classes = {gap_classes}")

    intervals_init = create_intervals(df)

    df_init = df.copy()
    gap_percent = round(df_init[col_target].isna().mean() * 100)

    # ======= ОЦЕНКА ==================

    df_without_nan = df_init.dropna()
    intervals_without_nan = create_intervals(df_without_nan)

    df_without_nan[col_time] = pd.to_datetime(df_without_nan[col_time])

    df_without_nan = df_without_nan.sort_values(col_time)

    start_date = df_without_nan[col_time].iloc[3]
    start_date = pd.to_datetime(start_date)


    df_orig, df_test_with_gaps, drop_indexes = AIM_create_test_nan(
        initial_df=df_without_nan, test_start_date=start_date,
        percent_gaps=gap_percent, col_time=col_time, target_value=col_target, intervals=intervals_init, gap_classes=gap_classes,
    )

    print(df_orig[df_orig["is_droped"]])

    df_only_gaps = df_orig.loc[drop_indexes]

    df_orig[col_time] = pd.to_datetime(df_orig[col_time])
    df_test_with_gaps[col_time] = pd.to_datetime(df_test_with_gaps[col_time])

    df_orig = df_orig.set_index(col_time)
    df_test_with_gaps = df_test_with_gaps.set_index(col_time)

    df_only_gaps.set_index(col_time)
    df_only_gaps = df_only_gaps.set_index(col_time)

    drop_cols = ["interval", "is_droped", "gap_classes"]

    df_filled_HDIRT = HDIRT_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_HDIRT = df_filled_HDIRT.rename(columns={col_target: f"HDIRT_{col_target}"})
    df_filled_HDIRT = df_filled_HDIRT.drop(columns=drop_cols)

    df_filled_MEAN = MEAN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_MEAN = df_filled_MEAN.rename(columns={col_target: f"MEAN_{col_target}"})
    df_filled_MEAN = df_filled_MEAN.drop(columns=drop_cols)

    df_filled_MEAN_BETWEEN = MEAN_BETWEEN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_MEAN_BETWEEN = df_filled_MEAN_BETWEEN.rename(columns={col_target: f"MEAN_BETWEEN_{col_target}"})
    df_filled_MEAN_BETWEEN = df_filled_MEAN_BETWEEN.drop(columns=drop_cols)

    df_filled_SKNN = SKNN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_SKNN = df_filled_SKNN.rename(columns={col_target: f"SKNN_{col_target}"})
    df_filled_SKNN = df_filled_SKNN.drop(columns=drop_cols)

    df_filled_KNN = KNN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_KNN = df_filled_KNN.rename(columns={col_target: f"KNN_{col_target}"})
    df_filled_KNN = df_filled_KNN.drop(columns=drop_cols)

    df_filled_LR = LR_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_LR = df_filled_LR.rename(columns={col_target: f"LR_{col_target}"})
    df_filled_LR = df_filled_LR.drop(columns=drop_cols)

    df_filled_LAST = LAST_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_LAST = df_filled_LAST.rename(columns={col_target: f"LAST_{col_target}"})
    df_filled_LAST = df_filled_LAST.drop(columns=drop_cols)

    df_filled_MEDIAN = MEDIAN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_MEDIAN = df_filled_MEDIAN.rename(columns={col_target: f"MEDIAN_{col_target}"})
    df_filled_MEDIAN = df_filled_MEDIAN.drop(columns=drop_cols)

    df_filled_SMEAN = SMEAN_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_SMEAN = df_filled_SMEAN.rename(columns={col_target: f"SMEAN_{col_target}"})
    df_filled_SMEAN = df_filled_SMEAN.drop(columns=drop_cols)

    df_filled_LINTER = LINTER_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_LINTER = df_filled_LINTER.rename(columns={col_target: f"LINTER_{col_target}"})
    df_filled_LINTER = df_filled_LINTER.drop(columns=drop_cols)

    df_filled_XGB = XGB_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_XGB = df_filled_XGB.rename(columns={col_target: f"XGB_{col_target}"})
    df_filled_XGB = df_filled_XGB.drop(columns=drop_cols)

    df_filled_SFLXGB = SFLXGB_imputation(df=df_test_with_gaps, col_target=col_target)
    df_filled_SFLXGB = df_filled_SFLXGB.rename(columns={col_target: f"SFLXGB_{col_target}"})
    df_filled_SFLXGB = df_filled_SFLXGB.drop(columns=drop_cols)


    df_filled = pd.concat(
        [
            df_orig,
            df_filled_HDIRT,
            df_filled_MEAN,
            df_filled_MEAN_BETWEEN,
            df_filled_SKNN,
            df_filled_KNN,
            df_filled_LR,
            df_filled_LAST,
            df_filled_MEDIAN,
            df_filled_SMEAN,
            df_filled_LINTER,
            df_filled_XGB,
            df_filled_SFLXGB
        ],
        axis=1
    )

    df = df_filled[df_filled["is_droped"] == True]

    print(f"df_filled")
    print(df)


    methods = df.columns.tolist()
    methods = [m for m in methods if m not in {col_target, "interval", "is_droped", "gap_classes"}]
    methods = [m.replace(f"_{col_target}", "") for m in methods]

    metrics = calc_metrics_by_class(df=df, target_col=col_target, methods=methods)

    df_metrics = pd.DataFrame(metrics)

    best_methods = (
        df_metrics.loc[df_metrics.groupby("class")["mape"].idxmin()]
        .reset_index(drop=True)
    )
    # ======= ЗАПОЛНЕНИЕ ==================

    gap_intervals = get_gap_intervals(df_init, col_target)

    df_init["gap_classes"] = None
    for start, end in gap_intervals:
        length = end - start
        cls = get_gap_class(length, gap_classes)
        df_init.iloc[start:end+1, df_init.columns.get_loc("gap_classes")] = cls


    df_init = df_init.set_index(col_time)

    df_init_filled_HDIRT = HDIRT_imputation(df=df_init, col_target=col_target)
    df_init_filled_HDIRT = df_init_filled_HDIRT.rename(columns={col_target: f"HDIRT_{col_target}"})
    df_init_filled_HDIRT = df_init_filled_HDIRT.drop(columns=drop_cols)

    df_init_filled_MEAN = MEAN_imputation(df=df_init, col_target=col_target)
    df_init_filled_MEAN = df_init_filled_MEAN.rename(columns={col_target: f"MEAN_{col_target}"})
    df_init_filled_MEAN = df_init_filled_MEAN.drop(columns=drop_cols)

    df_init_filled_MEAN_BETWEEN = MEAN_BETWEEN_imputation(df=df_init, col_target=col_target)
    df_init_filled_MEAN_BETWEEN = df_init_filled_MEAN_BETWEEN.rename(columns={col_target: f"MEAN_BETWEEN_{col_target}"})
    df_init_filled_MEAN_BETWEEN = df_init_filled_MEAN_BETWEEN.drop(columns=drop_cols)

    df_init_filled_SKNN = SKNN_imputation(df=df_init, col_target=col_target)
    df_init_filled_SKNN = df_init_filled_SKNN.rename(columns={col_target: f"SKNN_{col_target}"})
    df_init_filled_SKNN = df_init_filled_SKNN.drop(columns=drop_cols)

    df_init_filled_KNN = KNN_imputation(df=df_init, col_target=col_target)
    df_init_filled_KNN = df_init_filled_KNN.rename(columns={col_target: f"KNN_{col_target}"})
    df_init_filled_KNN = df_init_filled_KNN.drop(columns=drop_cols)

    df_init_filled_LR = LR_imputation(df=df_init, col_target=col_target)
    df_init_filled_LR = df_init_filled_LR.rename(columns={col_target: f"LR_{col_target}"})
    df_init_filled_LR = df_init_filled_LR.drop(columns=drop_cols)

    df_init_filled_LAST = LAST_imputation(df=df_init, col_target=col_target)
    df_init_filled_LAST = df_init_filled_LAST.rename(columns={col_target: f"LAST_{col_target}"})
    df_init_filled_LAST = df_init_filled_LAST.drop(columns=drop_cols)

    df_init_filled_MEDIAN = MEDIAN_imputation(df=df_init, col_target=col_target)
    df_init_filled_MEDIAN = df_init_filled_MEDIAN.rename(columns={col_target: f"MEDIAN_{col_target}"})
    df_init_filled_MEDIAN = df_init_filled_MEDIAN.drop(columns=drop_cols)

    df_init_filled_SMEAN = SMEAN_imputation(df=df_init, col_target=col_target)
    df_init_filled_SMEAN = df_init_filled_SMEAN.rename(columns={col_target: f"SMEAN_{col_target}"})
    df_init_filled_SMEAN = df_init_filled_SMEAN.drop(columns=drop_cols)

    df_init_filled_LINTER = LINTER_imputation(df=df_init, col_target=col_target)
    df_init_filled_LINTER = df_init_filled_LINTER.rename(columns={col_target: f"LINTER_{col_target}"})
    df_init_filled_LINTER = df_init_filled_LINTER.drop(columns=drop_cols)

    df_init_filled_XGB = XGB_imputation(df=df_init, col_target=col_target)
    df_init_filled_XGB = df_init_filled_XGB.rename(columns={col_target: f"XGB_{col_target}"})
    df_init_filled_XGB = df_init_filled_XGB.drop(columns=drop_cols)

    df_init_filled_SFLXGB = SFLXGB_imputation(df=df_init, col_target=col_target)
    df_init_filled_SFLXGB = df_init_filled_SFLXGB.rename(columns={col_target: f"SFLXGB_{col_target}"})
    df_init_filled_SFLXGB = df_init_filled_SFLXGB.drop(columns=drop_cols)

    df_result = pd.concat(
        [
            df_init,
            df_init_filled_HDIRT,
            df_init_filled_MEAN,
            df_init_filled_MEAN_BETWEEN,
            df_init_filled_SKNN,
            df_init_filled_KNN,
            df_init_filled_LR,
            df_init_filled_LAST,
            df_init_filled_MEDIAN,
            df_init_filled_SMEAN,
            df_init_filled_LINTER,
            df_init_filled_XGB,
            df_init_filled_SFLXGB
        ],
        axis=1
    )

    df_result = apply_best_method_by_class(
        df_result=df_result,
        best_methods=best_methods,
        target_col=col_target
    )

    cols_to_chose = [col_target, "interval", "is_droped", f"AIM_{col_target}"]
    df_result = df_result[cols_to_chose]

    return df_result
