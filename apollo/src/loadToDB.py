import logging
import pandas as pd
import oandapy as opy
from datetime import datetime as dt
from sqlalchemy import create_engine


def get_instrument_history(instrument,
              granularity,
              start,
              end,
              candleformat,
              freq,
              trading=False):
    """
    Oanda FX historical data

    Args:
        instrument (str): Objective instrument
        granularity (str): Time Window
        start (str): First day
        end (str): Last day
        candleformat (str): 'bidask' or 'midpoint'
        freq (str): Timeframe
        trading (bool): If trading
    Returns:tqdm
        df (DataFrame)
    """

    oanda = opy.API(environment='live')

    dates = pd.date_range(start=start, end=end, freq=freq)
    df = pd.DataFrame()

    if trading:
        data = oanda.get_history(instrument=instrument,
                                     candleFormat=candleformat,
                                     since=start,
                                     granularity=granularity)
        df = pd.DataFrame(data['candles'])
    else:

        for i in range(0, len(dates) - 1):
            d1 = str(dates[i]).replace(' ', 'T')
            d2 = str(dates[i+1]).replace(' ', 'T')
            try:
                data = oanda.get_history(instrument=instrument,
                                         candleFormat=candleformat,
                                         start=d1,
                                         end=d2,
                                         granularity=granularity)

                df = df.append(pd.DataFrame(data['candles']))
            except:
                pass

    date = pd.DatetimeIndex(df['time'])
    df['time'] = date

    return df

def load_instrument_history():

    candleformat = 'bidask'
    instrument = 'USD_JPY'
    granularity = 'H1'
    start = '2008-01-01'
    end = str(dt.now())[:12]
    freq = '5D'
    trading = False

    data = get_instrument_history(instrument,
                      granularity,
                      start,
                      end,
                      candleformat,
                      freq,
                      trading=trading)

    engine = create_engine('postgresql://dud:dud@localhost:5432/duddb')
    data.to_sql('dudtablename', engine)
