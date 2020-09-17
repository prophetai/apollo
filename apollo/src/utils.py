import pandas as pd
import numpy as np
import oandapy as opy
import logging
import os
from datetime import datetime as dt
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

def get_forex(instrument,
              instruments,
              granularity,
              start,
              end,
              candleformat,
              freq,
              trading=False):
    """
    Obtiene datos FX de Oanda para los instrumentos que elijamos

    Args:
        instrument (str): Instrumento a predecir
        instruments (list): Divisas
        granularity (str): Time Window
        start (str): Primer día
        end (str): último día
        candleformat (str): 'bidask' o 'midpoint'
        freq (str): Timeframe
        trading (bool): Si estamos en producción
    Returns:
        df (DataFrame)
    """

    oanda = opy.API(environment='live')
    divs = {}

    for j in instruments:
        logging.info(j)
        # Extraemos datos cada 2 días (por simplicidad)
        d1 = start
        d2 = end
        dates = pd.date_range(start=d1, end=d2, freq=freq)
        df = pd.DataFrame()

        if trading:
            data = oanda.get_history(instrument=j,
                                         candleFormat=candleformat,
                                         since=d1,
                                         granularity=granularity)
            df = pd.DataFrame(data['candles'])
        else:
            pbar = tqdm(total=len(dates) - 1)

            for i in range(0, len(dates) - 1):
                # Oanda toma las fechas en este formato
                d1 = str(dates[i]).replace(' ', 'T')
                d2 = str(dates[i+1]).replace(' ', 'T')
                try:
                    # Concatenamos cada día en el dataframe

                    data = oanda.get_history(instrument=j,
                                             candleFormat=candleformat,
                                             start=d1,
                                             end=d2,
                                             granularity=granularity)

                    df = df.append(pd.DataFrame(data['candles']))
                    pbar.update(1)
                except:
                    pass

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


def calculate_difference(dat, log=True, drop=False):
    """
    Calcula diferencia entre el valor actual y el pasado

    Args:
        df (DataFrame): Datos
        log (bool): Diferencia logaritmica
    Returns:
        df (DataFrame): Datos con diferencias
    """
    df = dat.copy()
    drop_vars = []
    for i in df.columns:
        try:
            if log:
                df['LogDiff ' + i] = np.log(df[i]).diff(1)
            else:
                df['Diff ' + i] = df[i].diff(1)
            drop_vars.append(i)

        except Exception as e:
            logging.warning(e)

    if drop:
        df = df.drop(drop_vars, axis=1)

    return df, drop_vars

def setup_data(dat,
               instrument='USD_JPY',
               pricediff=True,
               log=True,
               trading=False):
    """
    Calcula diferencias y acomoda datos

    Args:
        dat (DataFrame): Datos
        instrument (str): Divisa
        pricediff (bool): Si queremos diferencias en precios
        log (bool): Si queremos transformación logarítmica
        trading (bool): Si estamos en producción
    Returns:
        df (DataFrame): Datos transformados y ajustados
    """

    df = dat.copy()
    date = '{}_date'.format(instrument)
    drops = [k for k in df.columns if date not in k and ('date' in k or 'complete' in k or 'time' in k)]
    df = df.drop(drops, axis=1)
    if trading == False:
        df = df[100:] # Falla en API
    df = df.reset_index(drop=True)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    drops = []
    if pricediff:
        df, drops = calculate_difference(df, log=log)

    high = instrument + '_highMid'
    low = instrument + '_lowMid'
    close = instrument + '_closeMid'
    hcdiff = 'Diff High-Close'
    cldiff = 'Diff Close-Low'
    hldiff = 'Diff High-Low'

    df[hcdiff] = df[high] - df[close]
    df[cldiff] = df[close] - df[low]
    df[hldiff] = df[high] - df[low]

    drops = [i for i in drops if i not in [date, hcdiff, cldiff, hldiff] and 'volume' not in i]
    #df = df.drop(drops, axis=1)
    df = df[1:]
    df[date] = df[date].astype(str)
    df[date] = df[date].str[:13]
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor