"""Utilities specific to photovoltaic generator scenarios."""

import pandas as pd
from datetime import datetime
from typing import Tuple, Set


def overlap(a: Tuple[pd.Timestamp, pd.Timestamp],
            b: Tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    """Determine whether two time intervals overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


def get_yearly_date_range(
        date: pd.Timestamp, num_of_days: int = 60,
        start: str = '2017-01-01', end: str = '2018-12-31'
        ) -> Set[pd.Timestamp]:
    """Get date range around a specific date."""

    hist_dates = pd.date_range(start=start, end=end, freq='D', tz='utc')
    hist_years = hist_dates.year.unique()
    hist_dates = set(hist_dates)

    # take given number of days before and after
    near_dates = set()
    for year in hist_years:
        year_date = datetime(year, date.month, date.day)

        near_dates = near_dates.union(set(pd.date_range(
            start=year_date - pd.Timedelta(num_of_days, unit='D'),
            periods=2 * num_of_days + 1, freq='D', tz='utc')
            ))

    return hist_dates.intersection(near_dates)
