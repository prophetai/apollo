import pandas as pd
import oandapy as opy
import logging
from tqdm import tqdm
from datetime import datetime as dt
from datetime import timedelta 
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
    print(d1,d2)
    dates = pd.date_range(start=d1, end=d2, freq=freq)
    dates = [f'{date}Z'.replace(' ','T') for date in dates]
    dates.append(f'{dt.now()}Z'.replace(' ','T').split('.')[0])

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
    else:
        dat = pd.DataFrame()
        #dates.insert(0,start)
        dates = list(dict.fromkeys(dates))
        pbar = tqdm(total=len(dates) - 1)
        for i in range(len(dates)-1):
            pbar.set_description_str(f'[{instrument}]Dates:{dates[i]}->{dates[i+1]}')
            fx_data = oanda.get_history(instrument=instrument,
                                       candleFormat=candleformat,
                                       start=dates[i],
                                       end=dates[i+1],
                                       granularity=granularity)
            fx_data = pd.DataFrame(fx_data['candles'])                           
            dat = dat.append(fx_data)
            pbar.update(1,)
        pbar.close()
        date = pd.DatetimeIndex(dat['time'])
        dat['date'] = date

    return dat
