import logging
import sys

import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """
    Reads data from input file and provides them as Pandas DataFrame.
    """
    try:
        df = pd.read_csv(
            path,
            sep=" ",
            header=None,
            names=["date", "count"],
            dtype="uint16",
            parse_dates=["date"],
        )
    except FileNotFoundError as e:
        logging.warning("Input data file not found.")
        raise e

    return df


def total(df: pd.DataFrame) -> int:
    """
    The number of cars seen in total
    """
    return df["count"].sum()


def day_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Totals per each observed day.
    """
    return df.groupby(df["date"].dt.date).sum()


def top3(df: pd.DataFrame) -> pd.DataFrame:
    """
    The top 3 half hours with most cars.
    """
    return df.sort_values("count", ascending=False).head(3)


def period_least(df: pd.DataFrame) -> str:
    """
    The 1.5 hour period with least cars (i.e. 3 contiguous half hour records).
    """
    # get rolling window sums
    rolling_sums = df.rolling(on="date", window="90min", min_periods=3).sum()
    # raise an error if no 3 contiguous half hour records were found
    if rolling_sums.dropna().empty:
        raise ValueError("Dataset doesn't contain any 3 contiguous half hour records.")

    # get index of the row with smallest window sum
    loc = rolling_sums.sort_values("count").index[0]

    # use the index to get all records in the window
    return df.iloc[loc - 2 : loc + 1]


def format_df(df: pd.DataFrame) -> str:
    """
    Converts results DataFrame to formatted string output.
    """
    formatted = df.reset_index().apply(
        lambda r: f"{r['date'].isoformat()} {r['count']}", axis=1
    )
    return "\n".join(formatted.tolist())


def main():
    if len(sys.argv) < 2:
        raise ValueError("Traffic data file not provided.")

    df = read_data(sys.argv[1])

    results = {
        "TOTAL": total(df),
        "DAY TOTALS": day_totals(df),
        "TOP 3": top3(df),
        "PERIOD LEAST": period_least(df),
    }

    # print out all results
    for title, result in results.items():
        # present DataFrames in standardised format
        if isinstance(result, pd.DataFrame):
            result = format_df(result)

        print(title, "-" * len(title), result, sep="\n", end="\n\n")


if __name__ == "__main__":
    main()
