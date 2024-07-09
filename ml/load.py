import os

import pandas as pd


def clean_data(df):
    cutoff_date = df[df[df.columns[1]].isnull()].index[0]
    df = df[df.index < cutoff_date].copy()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Convert all columns to float and reset the index
    df = (
        df.astype(float)
        .reset_index()
        .set_index("date")[
            [
                "lo_10",
                "dec_2",
                "dec_3",
                "dec_4",
                "dec_5",
                "dec_6",
                "dec_7",
                "dec_8",
                "dec_9",
                "hi_10",
            ]
        ]
        .melt(ignore_index=False, var_name="ticker", value_name="return_1d")
    )

    df["month"] = df.index.to_period("M")
    return df


def process_files(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):  # assuming the files are CSVs
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            cleaned_df = clean_data(df)
            output_file_path = os.path.join(output_directory, filename)
            cleaned_df.to_csv(output_file_path)


if __name__ == "__main__":
    input_directory = "/Users/noahforougi/research/portfolio_optimization/data/raw/"
    output_directory = "/Users/noahforougi/research/portfolio_optimization/data/clean/"
    process_files(input_directory, output_directory)
