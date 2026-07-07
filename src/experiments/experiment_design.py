"""
pdm run src/experiments/experiment_design.py
"""
import os
import pandas as pd
from src.data.data_config import datasets_csv_dict

home_path = os.getcwd()
export_path = os.path.join(home_path, "export")

datasets_to_test = ["russia_amur_region", "Daily_Climate", "Temperature_in_Celsius", "Istanbul_Traffic_Index", "russia_elista"]

datasets_to_test = ["russia_amur_region"]

# prc_list = [10, 30, 50, 70]
prc_list = [3]


def create_experiment_design(experiment_path):
    rows = []

    for dataset in datasets_to_test:
        csv = datasets_csv_dict[dataset]["csv_link"]
        col_time = datasets_csv_dict[dataset]["col_time"]
        col_target = datasets_csv_dict[dataset]["col_target"]
        for prc in prc_list:
            path_to_save = os.path.join(
                experiment_path,
                "results",
                dataset,
                str(prc),
            )

            rows.append({
                "dataset": dataset,
                "csv_link": csv,
                "col_time": col_time,
                "col_target": col_target,
                "gap_prc": prc,
                "path_to_save": path_to_save,

                })

    return pd.DataFrame(rows)
