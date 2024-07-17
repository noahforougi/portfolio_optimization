import os

import pandas as pd

import config
import utils


def clean_data(df):
    cutoff_date = df[df[df.columns[1]].isnull()].index[0]
    df = df[df.index < cutoff_date].copy()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df.astype(float).reset_index()


def main():
    for k, v in config.COLUMN_DICT.items():
        df = utils.read_s3_file(f"raw/{k}")
        df_clean = clean_data(df)
        df_clean = df_clean[["date"] + v[1]]
        clean_file_key = f"clean/{v[0]}.csv"
        utils.write_s3_file(df_clean, clean_file_key)
        print(f"Processed and saved {k} to {v[0]}.csv")
