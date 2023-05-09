"""Utilities specific to photovoltaic generator scenarios."""

import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from astral import LocationInfo
from astral.sun import sunrise, sunset
from typing import Tuple

def get_asset_transition_hour_info(
        loc: LocationInfo, date: datetime.date,
        sunrise_delay_in_minutes: int = 0, sunset_delay_in_minutes: int = 0
        ) -> dict:
    """Get asset sunrise and sunset horizon and fraction."""

    # calculate sun related info in local time, then convert into utc
    sun_rise = sunrise(loc.observer, date = date,
         tzinfo = pytz.timezone(loc.timezone)).astimezone(timezone.utc)

    sun_set = sunset(loc.observer, date = date,
         tzinfo=pytz.timezone(loc.timezone)).astimezone(timezone.utc)

    if sun_set < sun_rise:
        sun_set = sunset(loc.observer, date = (date + timedelta(days=1)).date(),
         tzinfo=pytz.timezone(loc.timezone)).astimezone(timezone.utc)

    sun_info = {'sunrise': sun_rise, 'sunset': sun_set}

    sunrise_delay_time = pd.Timedelta(sunrise_delay_in_minutes, unit='min')
    sunset_delay_time = pd.Timedelta(sunset_delay_in_minutes, unit='min')

    # Sunrise
    sunrise_time = pd.to_datetime(sun_info['sunrise'])
    sunrise_timestep = (sunrise_time + sunrise_delay_time).floor('H')
    sunrise_active_min = sun_info['sunrise'].minute + sunrise_delay_in_minutes
    sunrise_active_min = 60 - sunrise_active_min % 60

    # Sunset
    sunset_time = pd.to_datetime(sun_info['sunset'])
    sunset_timestep = (sunset_time - sunset_delay_time).floor('H')
    sunset_active_min = (sun_info['sunset'].minute
                         - sunset_delay_in_minutes) % 60

    return {'sunrise': {'time': sunrise_time, 'timestep': sunrise_timestep,
                        'active': sunrise_active_min},
            'sunset': {'time': sunset_time, 'timestep': sunset_timestep,
                       'active': sunset_active_min}}
