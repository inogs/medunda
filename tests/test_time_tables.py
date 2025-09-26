from datetime import datetime

from medunda.tools import time_tables


def test_get_next_month_normal():
    date = datetime(2023, 5, 15)
    result = time_tables.get_next_month(date)
    assert result == datetime(2023, 6, 1)


def test_get_next_month_december():
    date = datetime(2023, 12, 31)
    result = time_tables.get_next_month(date)
    assert result == datetime(2024, 1, 1)


def test_get_next_year():
    date = datetime(2022, 7, 10)
    result = time_tables.get_next_year(date)
    assert result == datetime(2023, 1, 1)


def test_split_by_month_single_month():
    start = datetime(2023, 5, 2)
    end = datetime(2023, 5, 12)
    intervals = time_tables.split_by_month(start, end)
    assert len(intervals) == 1
    assert intervals[0][0] == start
    assert intervals[0][1] == datetime(2023, 5, 12)


def test_split_by_month_multiple_months():
    start = datetime(2023, 4, 15)
    end = datetime(2023, 6, 10)
    intervals = time_tables.split_by_month(start, end)
    assert len(intervals) == 3
    assert intervals[0][0] == start
    assert intervals[1][0] == datetime(2023, 5, 1)
    assert intervals[2][0] == datetime(2023, 6, 1)
    assert intervals[2][1] == end


def test_split_by_month_between_years():
    start = datetime(2023, 11, 15)
    end = datetime(2024, 2, 7)
    intervals = time_tables.split_by_month(start, end)
    assert len(intervals) == 4
    assert intervals[0][0] == start
    assert intervals[1][0] == datetime(2023, 12, 1)
    assert intervals[2][0] == datetime(2024, 1, 1)
    assert intervals[3][0] == datetime(2024, 2, 1)
    assert intervals[3][1] == end


def test_split_by_year_single_year():
    start = datetime(2022, 2, 12)
    end = datetime(2022, 11, 29)
    intervals = time_tables.split_by_year(start, end)
    assert len(intervals) == 1
    assert intervals[0][0] == start
    assert intervals[0][1] == end


def test_split_by_year_multiple_years():
    start = datetime(2021, 6, 1)
    end = datetime(2023, 2, 1)
    intervals = time_tables.split_by_year(start, end)
    assert len(intervals) == 3
    assert intervals[0][0] == start
    assert intervals[1][0] == datetime(2022, 1, 1)
    assert intervals[-1][1] == end
