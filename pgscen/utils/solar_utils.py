import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime
from operator import itemgetter

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

def get_activate_assets(sun_type,meta_df,time_period_start,time_period_in_minutes=60,delay_in_minutes=10):
    """
    """

    local_date = time_period_start.tz_convert('US/Central').date()
    if sun_type == 'sunrise':
        dt = time_period_start+pd.Timedelta(60-delay_in_minutes,unit='min')
    elif sun_type == 'sunset':
        dt = time_period_start+pd.Timedelta(delay_in_minutes,unit='min')
    
    asset_list = []
    for _,row in meta_df.iterrows():
        site = row['site_ids']
        lat = row['latitude']
        lon = row['longitude']
        loc = LocationInfo(site,'Texas','USA',lat,lon)
        s = sun(loc.observer,date=local_date)
        
        if sun_type == 'sunrise' and pd.to_datetime(s['sunrise']) < dt:
            asset_list.append(site)
        elif sun_type == 'sunset' and pd.to_datetime(s['sunset']) > dt:
            asset_list.append(site)

    return sorted(asset_list)


def get_solar_hist_dates(date, meta_df, hist_start, hist_end,
                         time_range_in_minutes=15):

    hist_dates = pd.date_range(start=hist_start, end=hist_end,
                               freq='D', tz='utc')
    hist_years = hist_dates.year.unique()
    hist_dates = set(hist_dates)

    # Take 60 days before and after
    near_dates = set()
    for year in hist_years:
        year_date = datetime(year,date.month, date.day)
        near_dates = near_dates.union(
            set(pd.date_range(start=year_date - pd.Timedelta(60, unit='D'),
                              periods=121, freq='D', tz='utc'))
            )
    hist_dates = hist_dates.intersection(near_dates)

    asset_locs = {site: LocationInfo(site, 'Texas', 'USA', lat, lon).observer
                  for site, lat, lon in zip(meta_df.index,
                                            meta_df.latitude,
                                            meta_df.longitude)}

    asset_suns = {site: sun(loc, date=date)
                  for site, loc in asset_locs.items()}

    for site in asset_suns:
        for sun_time in ['sunrise', 'sunset']:
            asset_suns[site][sun_time] = datetime.combine(
                datetime.min,
                pd.to_datetime(asset_suns[site][sun_time]).tz_convert(
                    'US/Central').time()
                )

    cur_rises = pd.Series({site: s['sunrise']
                           for site, s in asset_suns.items()})
    cur_sets = pd.Series({site: s['sunset'] for site, s in asset_suns.items()})

    hist_suns = {site: {hist_date: sun(loc, date=hist_date)
                        for hist_date in hist_dates}
                 for site, loc in asset_locs.items()}

    for site in asset_suns:
        for hist_date in hist_dates:
            for sun_time in ['sunrise', 'sunset']:
                hist_suns[site][hist_date][sun_time] = datetime.combine(
                    datetime.min,
                    pd.to_datetime(
                        hist_suns[site][hist_date][sun_time]).tz_convert(
                        'US/Central'
                        ).time()
                    )

    sun_df = pd.DataFrame(hist_suns)
    sunrise_df = sun_df.applymap(itemgetter('sunrise'))
    sunset_df = sun_df.applymap(itemgetter('sunset'))

    max_diff = pd.Timedelta(time_range_in_minutes, unit='min')
    rise_diffs = (sunrise_df - cur_rises).abs() <= max_diff
    set_diffs = (sunset_df - cur_sets).abs() <= max_diff
    hist_stats = (rise_diffs & set_diffs).all(axis=1)

    return {hist_time
            for hist_time, hist_stat in hist_stats.iteritems() if hist_stat}


# def get_hist_dates(scen_start_time,start,end,meta_df,sun='both',range_in_seconds=900):
#     """
#     Find historical dates of solar assets based on the following criteria:
#         1. the intersection of dates for all solar assets that have sunrise time within a time range to the scenario date
#         2. the intersection of dates for all solar assets that have sunset time within a time range to the scenario date.
#         3. intersection of 1 and 2 above.

#     :param scen_start_time: scenario start time
#     :type scen_start_time: pandas Timestamp
#     :param start: historical start date
#     :type start: pandas Timestamp
#     :param end: historical end date
#     :type end: pandas Timestamp
#     :param meta_df: solar meta data
#     :type meta_df: pandas DataFrame
#     :param sun: criteria to select historical dates, can be ``sunrise``, ``sunset`` or ``both``.
#     :type sun: str
#     :param range_in_seconds: 
#     :type range_in_seconds:
#     """
    
#     hist_dates = set(pd.date_range(start=start,end=end,freq='D'))
    
#     # Take 60 days before and after scen_start_time
#     s1 = set(pd.date_range(start=scen_start_time-pd.Timedelta(60,unit='D'),periods=60,freq='D'))
#     s2 = set(pd.date_range(start=scen_start_time+pd.Timedelta(1,unit='D'),periods=60,freq='D'))
#     s3 = set(pd.date_range(start=scen_start_time-pd.Timedelta(425,unit='D'),periods=121,freq='D'))
#     s4 = set(pd.date_range(start=scen_start_time+pd.Timedelta(305,unit='D'),periods=121,freq='D'))
#     hist_dates = hist_dates.intersection(s1.union(s2.union(s3.union(s4))))
    
#     for _,row in meta_df.iterrows():
#         site = row['site_ids']
#         lon = row['longitude']
#         lat = row['latitude']

#         loc = LocationInfo(site,'Texas','USA',lat,lon)
#         rdates,sdates = get_solar_dates_within_range(loc,scen_start_time,start,end,range_in_seconds=range_in_seconds)
        
#         if sun == 'both':
#             hist_dates = hist_dates.intersection(set(rdates).intersection(set(sdates)))
#         elif sun == 'rise':
#             hist_dates = hist_dates.intersection(set(rdates))
#         elif sun == 'set':
#             hist_dates = hist_dates.intersection(set(sdates))
#         else:
#             raise(RuntimeError('input ``sun`` has to be one of ``rise``, ``set`` or ``both``, got {}'.format(sun)))
        
#     hist_dates = [dt+pd.Timedelta(6,unit='H') for dt in hist_dates]
    
#     return hist_dates

