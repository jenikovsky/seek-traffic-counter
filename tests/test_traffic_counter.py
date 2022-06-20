import os
import re

import pandas as pd
import pytest

os.sys.path.append(os.path.join(os.curdir, "traffic_counter"))
import traffic_counter as tc


@pytest.fixture
def data():
    return tc.read_data("data/traffic.txt")


@pytest.fixture
def one_record():
    return pd.DataFrame({"date": [pd.to_datetime("2021-01-01T01:00:00")], "count": [0]})


def test_read_data():
    with pytest.raises(FileNotFoundError):
        tc.read_data("no-file")


def test_total(data):
    assert tc.total(data) == 398


def test_day_totals(data):
    df = tc.day_totals(data)
    assert df.shape[0] == 4
    assert df["count"].iloc[2] == 134


def test_top3(data, one_record):
    assert tc.top3(data)["count"].iloc[1] == 42
    # if less than 3 records in dataset, return maximum available
    assert tc.top3(one_record).shape[0] == 1


def test_period_least(data, one_record):
    df = tc.period_least(data)
    assert df.shape[0] == 3
    assert df["count"].sum() == 31

    # no 3 contiguous records in dataset
    with pytest.raises(ValueError):
        tc.period_least(one_record)


def test_format(data):
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s\d+"
    assert re.match(pattern, tc.format_df(data))
