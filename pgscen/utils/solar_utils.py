import pandas as pd
from astral import LocationInfo
from astral.sun import sun
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
    
def get_similar_solar_dates(loc,dt,start,end,time_range_in_minutes=15):
    """
    Find dates during when a give location's sunrise and sunset times 
    are within a time range of a datetime.

    :param loc: location in question
    :type loc: astral.LocationInfo
    :param dt: datetime in question
    :type dt: pandas Timestamp
    :param start: starting date of interest
    :type start: pandas Timestamp
    :param end: end date of interest
    :type end: pandas Timestamp
    :param range_in_seconds: time range in second
    :type range_in_seconds: int

    """
    s = sun(loc.observer,date=dt)
    current_sunrise_time = datetime.combine(datetime.min,pd.to_datetime(s['sunrise']).tz_convert('US/Central').time())
    current_sunset_time = datetime.combine(datetime.min,pd.to_datetime(s['sunset']).tz_convert('US/Central').time())

    sunrise_dates,sunset_dates = [],[]

    for date in pd.date_range(start=start,end=end,freq='D',tz='utc'):
        s = sun(loc.observer,date=date)    
        sunrise_time = datetime.combine(datetime.min,pd.to_datetime(s['sunrise']).tz_convert('US/Central').time())
        sunset_time = datetime.combine(datetime.min,pd.to_datetime(s['sunset']).tz_convert('US/Central').time())
            
        # Sunrise
        if sunrise_time<=current_sunrise_time:
            diff = current_sunrise_time-sunrise_time
        else:
            diff = sunrise_time-current_sunrise_time
            
        if diff<=pd.Timedelta(time_range_in_minutes,unit='min'):
            sunrise_dates.append(date)

        # Sunset
        if sunset_time<=current_sunset_time:
            diff = current_sunset_time-sunset_time
        else:
            diff = sunset_time-current_sunset_time
            
        if diff<=pd.Timedelta(time_range_in_minutes,unit='min'):
            sunset_dates.append(date)

    return sunrise_dates,sunset_dates

def get_solar_hist_dates(date,meta_df,hist_start,hist_end,time_range_in_minutes=15):

    hist_dates = pd.date_range(start=hist_start,end=hist_end,freq='D',tz='utc')
    hist_years = hist_dates.year.unique()
    hist_dates = set(hist_dates)

    # Take 60 days before and after
    near_dates = set()
    for year in hist_years:
        year_date = datetime(year,date.month,date.day)
        near_dates = near_dates.union(set(pd.date_range(start=year_date-pd.Timedelta(60,unit='D'),periods=121,freq='D',tz='utc')))
    hist_dates = hist_dates.intersection(near_dates)

    # Eliminate dates when sunrise and sunset times are different
    for _,row in meta_df.iterrows():
        site = row['site_ids']
        lon = row['longitude']
        lat = row['latitude']

        loc = LocationInfo(site,'Texas','USA',lat,lon)
        sunrise_dates,sunset_dates = get_similar_solar_dates(loc,date,hist_start,hist_end,time_range_in_minutes=time_range_in_minutes)

        hist_dates = hist_dates.intersection(set(sunrise_dates).intersection(set(sunset_dates)))

    return hist_dates

    

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

