import pandas as pd
import oandapy as opy
import logging
from tqdm import tqdm
from datetime import datetime as dt

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
    divs = {}

    for j in instruments:
        logging.info(j)
        d1 = start
        d2 = end
        dates = pd.date_range(start=d1, end=d2, freq=freq)
        dates = [str(date) for date in dates]
        dates.append(str(dt.now()))
        df = pd.DataFrame()

        if trading:
            data = oanda.get_history(instrument=j,
                                         candleFormat=candleformat,
                                         since=d1,
                                         granularity=granularity)
            df = pd.DataFrame(data['candles'])
            if df.empty:
                return df
        else:
            pbar = tqdm(total=len(dates) - 1)
            for i in range(0, len(dates) - 1):
                d1 = str(dates[i]).replace(' ', 'T')
                d2 = str(dates[i+1]).replace(' ', 'T')
                try:
                    logging.info(f'Intervalo de fechas:{dates}')
                    data = oanda.get_history(instrument=j,
                                            candleFormat=candleformat,
                                            start=d1,
                                            end=d2,
                                            granularity=granularity)
                    pbar.update(1)
                    df = df.append(pd.DataFrame(data['candles']))
                except Exception as e:
                    logging.error(f'error:{e}')
                    logging.error(f'start:{d1}, end:{d2}')                    
            if df.empty:
                logging.info('empty data from OANDA')
                return df

        if trading == False:
            pbar.close()

        date = pd.DatetimeIndex(df['time'])
        df['date'] = date
        cols = [j + '_' + k for k in df.columns]
        df.columns = cols
        divs[j] = df

    dat = divs[instruments[0]]
    for i in instruments[1:]:
        join_id = [k for k in divs[i].columns if 'date' in k][0]
        dat = pd.merge(dat,
                       divs[i],
                       left_on=instrument + '_date',
                       right_on=join_id, how='left')

    return dat
