import pandas as pd
import oandapy as opy
import logging
from tqdm import tqdm
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor


def get_forex(instrument,
              instruments,
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
        instruments (list): All instruments
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
    fx_dfs = {}
    fx_list = []

    d1 = start
    d2 = end
    dates = pd.date_range(start=d1, end=d2, freq=freq)
    dates = [str(date) for date in dates]
    dates.append(str(dt.now()))

    if trading:
        with ThreadPoolExecutor() as executor:
            fx_data = {executor.submit(oanda.get_history, instrument=instrument,
                                       candleFormat=candleformat,
                                       since=d1,
                                       granularity=granularity): instrument for instrument in instruments}

        fx_data = [pd.DataFrame(data.result()['candles']) for data in fx_data]
        fx_dfs = dict(zip(instruments, fx_data))

        for instrument in instruments:
            date = pd.DatetimeIndex(fx_dfs[instrument]['time'])
            cols = [instrument + '_' +
                    k for k in fx_dfs[instrument].columns if k != 'time']
            cols = [f'{instrument}_date'] + cols
            fx_dfs[instrument].columns = cols
            fx_dfs[instrument]['date'] = date

        dat = fx_dfs[instruments[0]]
        for i in instruments[1:]:
            dat = pd.concat([dat, fx_dfs[i]], axis=1)
    return dat
