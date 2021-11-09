
import pandas as pd
from datetime import datetime


def overlap(a,b):
    """
    Determine whether two time intervals overlap

    :param a: (start,end) of the first date range
    :type a: a tuple of pandas Timestamp
    :param b: (start,end) of the first date range
    :type b: a tuple of pandas Timestamp
    """
    if a[0]<=b[1] and b[0]<=a[1]:
        return True
    else:
        return False


def get_yearly_date_range(date, num_of_days=60,
                          start='2017-01-01', end='2018-12-31'):
    """
    Get date range around a specific date
    """
    hist_dates = pd.date_range(start=start, end=end, freq='D', tz='utc')
    hist_years = hist_dates.year.unique()
    hist_dates = set(hist_dates)

    # Take 60 days before and after
    near_dates = set()
    for year in hist_years:
        year_date = datetime(year, date.month, date.day)
        near_dates = near_dates.union(set(pd.date_range(
            start=year_date - pd.Timedelta(num_of_days, unit='D'),
            periods=2 * num_of_days + 1, freq='D', tz='utc')
            ))

    return hist_dates.intersection(near_dates)
