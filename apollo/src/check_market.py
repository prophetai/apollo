import pandas as pd
import datetime
import pandas_market_calendars as mcal

def market_open():
    """
    Checks if markets are open

    Returns:
        True if markets are open
        False if markets are closed
    """
    currentDT = str(datetime.datetime.now().strftime("%Y-%m-%d"))
    futureDT = str(datetime.datetime.now() + datetime.timedelta(days=+2))[:10]
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    nyse = mcal.get_calendar('NYSE')
    today_nyse = nyse.schedule(start_date=dt_string, end_date=dt_string)
    open_nyse = nyse.open_at_time(today_nyse, pd.Timestamp(dt_string, tz='UTC'))
    lse = mcal.get_calendar('LSE')
    today_lse = lse.schedule(start_date=dt_string, end_date=dt_string)
    open_lse = lse.open_at_time(today_lse, pd.Timestamp(dt_string, tz='UTC'))
    jpx= mcal.get_calendar('JPX')
    today_jpx = jpx.schedule(start_date=currentDT, end_date=futureDT)
    open_jpx = jpx.open_at_time(today_jpx, pd.Timestamp(dt_string, tz='UTC'))

    print(f'\nChecking Market hours On: {dt_string} \nMarkets Open:')
    print(f'Japan: {open_jpx}, London: {open_lse}, NY: {open_nyse}')
    print(f'Trading time:{open_jpx or open_lse or open_nyse}')

    return (open_jpx or open_lse or open_nyse)

if __name__ == "__main__":
    market_open()