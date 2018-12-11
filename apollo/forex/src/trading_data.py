#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import oandapy as opy
import time
import logging
import os
from datetime import datetime as dt
from datetime import timedelta
from tqdm import tqdm
import time
from bs4 import BeautifulSoup

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import datetime
import pytz

import statsmodels.api as sm

from statsmodels.regression.linear_model import OLSResults

COMMASPACE = ', '


# Extraemos datos de precios (Oanda) para el instrumento que elijamos y de twitter (Base de datos de Prophets)

def get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading=False):
    """
    Obtiene datos de FX de Oanda para los instrumentos que elijamos

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
                except:
                    pass


        date = pd.DatetimeIndex(df['time'], tz='UTC')
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


# ### Ajuste de Datos

# In[3]:


##### Checar logaritmos de volumenes

def adjust_lags(dat, min_window=None, instrument='USD_JPY', pricediff=True, candleformat='midpoint', log=True, trading=False):
    """
    Ajusta intervalos de tiempo en rangos de una hora

    Args:
        dat (DataFrame): Datos
        instrument (str): Divisa
        min_window (int): De cuántos minutos es cada intervalo
        pricediff (bool): Si queremos diferencias en precios
        candleformat (str): ['bidask', 'midpoint']
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
    if min_window:
        step = int(60/min_window)
    if candleformat == 'bidask':
        if pricediff:
            if log:
                for i in df.columns:
                    try:
                        df['Diff ' + i] = np.log(df[i]).diff(1)
                        drops.append(i)
                    except Exception as e:
                        print(e)
            else:

                for i in df.columns:
                    try:
                        df['Diff ' + i] = df[i] - df[i].shift(1)
                        drops.append(i)
                    except Exception as e:
                        print(e)

            if min_window:

                open_bid = ['Diff openBid' + str(min_window*(i+1)) for i in range(step)]
                open_ask = ['Diff openAsk' + str(min_window*(i+1)) for i in range(step)]
                close_bid = ['Diff closeBid' + str(min_window*(i+1)) for i in range(step)]
                close_ask = ['Diff closeAsk' + str(min_window*(i+1)) for i in range(step)]
                low_bid = ['Diff lowBid' + str(min_window*(i+1)) for i in range(step)]
                low_ask = ['Diff lowAsk' + str(min_window*(i+1)) for i in range(step)]
                high_bid = ['Diff highBid' + str(min_window*(i+1)) for i in range(step)]
                high_ask = ['Diff highAsk' + str(min_window*(i+1)) for i in range(step)]
                volume = ['volume' + str(min_window*(i+1)) for i in range(step)]

                shifts = list(range(1,step+1))

                for v, ob, oa, cb, ca, lb, la, hb, ha, s in zip(volume,
                                                             open_bid,
                                                             open_ask,
                                                             close_bid,
                                                             close_ask,
                                                             low_bid,
                                                             low_ask,
                                                             high_bid,
                                                             high_ask,
                                                             shifts):
                    df[v] = df['volume'].shift(s)
                    df[ob] = df['Diff openBid'].shift(s)
                    df[oa] = df['Diff openAsk'].shift(s)
                    df[cb] = df['Diff closeBid'].shift(s)
                    df[ca] = df['Diff closeAsk'].shift(s)
                    df[lb] = df['Diff lowBid'].shift(s)
                    df[la] = df['Diff lowAsk'].shift(s)
                    df[hb] = df['Diff highBid'].shift(s)
                    df[ha] = df['Diff highAsk'].shift(s)

        else:

            if log:
                for i in df.columns:
                    try:
                        df[i] = np.log(df[i])
                    except Exception as e:
                        print(e)
            if min_window:
                open_bid = ['openBid' + str(min_window*(i+1)) for i in range(step)]
                open_ask = ['openAsk' + str(min_window*(i+1)) for i in range(step)]
                close_bid = ['closeBid' + str(min_window*(i+1)) for i in range(step)]
                close_ask = ['closeAsk' + str(min_window*(i+1)) for i in range(step)]
                low_bid = ['lowBid' + str(min_window*(i+1)) for i in range(step)]
                low_ask = ['lowAsk' + str(min_window*(i+1)) for i in range(step)]
                high_bid = ['highBid' + str(min_window*(i+1)) for i in range(step)]
                high_ask = ['highAsk' + str(min_window*(i+1)) for i in range(step)]
                volume = ['volume' + str(min_window*(i+1)) for i in range(step)]

                shifts = list(range(1,step+1))

                for v, ob, oa, cb, ca, lb, la, hb, ha, s in zip(volume,
                                                             open_bid,
                                                             open_ask,
                                                             close_bid,
                                                             close_ask,
                                                             low_bid,
                                                             low_ask,
                                                             high_bid,
                                                             high_ask,
                                                             shifts):
                    df[v] = df['volume'].shift(s)
                    df[ob] = df['openBid'].shift(s)
                    df[oa] = df['openAsk'].shift(s)
                    df[cb] = df['closeBid'].shift(s)
                    df[ca] = df['closeAsk'].shift(s)
                    df[lb] = df['lowBid'].shift(s)
                    df[la] = df['lowAsk'].shift(s)
                    df[hb] = df['highBid'].shift(s)
                    df[ha] = df['highAsk'].shift(s)
    else:

        if pricediff:
            if log:
                for i in df.columns:
                    try:
                        df['Diff ' + i] = np.log(df[i]).diff(1)
                        drops.append(i)
                    except Exception as e:
                        print(e)
            else:
                for i in df.columns:
                    try:
                        df['Diff ' + i] = df[i] - df[i].shift(1)
                        drops.append(i)
                    except Exception as e:
                        print(e)
            if min_window:
                open_ = ['Diff openMid' + str(min_window*(i+1)) for i in range(step)]
                close = ['Diff closeMid' + str(min_window*(i+1)) for i in range(step)]
                low = ['Diff lowMid' + str(min_window*(i+1)) for i in range(step)]
                high = ['Diff highMid' + str(min_window*(i+1)) for i in range(step)]
                volume = ['volume' + str(min_window*(i+1)) for i in range(step)]

                shifts = list(range(1,step+1))

                for v, o, c, l, h, s in zip(volume,
                                             open_,
                                             close,
                                             low,
                                             high,
                                             shifts):
                    df[v] = df['volume'].shift(s)
                    df[o] = df['Diff openMid'].shift(s)
                    df[c] = df['Diff closeMid'].shift(s)
                    df[l] = df['Diff lowMid'].shift(s)
                    df[h] = df['Diff highMid'].shift(s)

        else:

            if log:
                for i in df.columns:
                    try:
                        df[i] = np.log(df[i])
                    except Exception as e:
                        print(e)

            if min_window:

                open_ = ['openMid' + str(min_window*(i+1)) for i in range(step)]
                close = ['closeMid' + str(min_window*(i+1)) for i in range(step)]
                low = ['lowMid' + str(min_window*(i+1)) for i in range(step)]
                high = ['highMid' + str(min_window*(i+1)) for i in range(step)]
                volume = ['volume' + str(min_window*(i+1)) for i in range(step)]

                shifts = list(range(1,step+1))

                for v, o, c, l, h, s in zip(volume,
                                             open_,
                                             close,
                                             low,
                                             high,
                                             shifts):
                    df[v] = df['volume'].shift(s)
                    df[o] = df['openMid'].shift(s)
                    df[c] = df['closeMid'].shift(s)
                    df[l] = df['lowMid'].shift(s)
                    df[h] = df['highMid'].shift(s)
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
    df = df.drop(drops, axis=1)
    df = df[1:]
    if min_window:
        fake_drop = [i for i in df.columns if 'volume' in i or 'date' in i]
        df['High'] = df.drop(fake_drop, 1).max(axis=1)
        df['Low'] = df.drop(fake_drop, 1).min(axis=1)
        df['vol'] = df[volume].sum(axis=1)
        df = df[step+1:]
    df[date] = df[date].astype(str)
    if min_window:
        df['d2'] = df[date].str[14:]
        df = df[df['d2'] == '00:00+00:00']
        df = df.reset_index(drop=True)
        df = df.drop('d2', axis=1)
    df[date] = df[date].str[:13]
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


# # Trading

# ### Precios

# In[4]:


variables = {'Diff USD_JPY_closeMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'intercept',
  'Diff USD_JPY_closeMid',
  'Diff USD_JPY_highMid',
  'Diff USD_JPY_lowMid'],
 'Diff USD_JPY_highMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'intercept',
  'Diff USD_JPY_closeMid',
  'Diff USD_JPY_highMid',
  'Diff USD_JPY_lowMid'],
 'Diff USD_JPY_lowMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'intercept',
  'Diff USD_JPY_closeMid',
  'Diff USD_JPY_highMid',
  'Diff USD_JPY_lowMid']}

models = {}
models['Linreg_Diff USD_JPY_closeMid'] = OLSResults.load('../models/Linreg_close.h5')
models['Linreg_Diff USD_JPY_highMid'] = OLSResults.load('../models/Linreg_high.h5')
models['Linreg_Diff USD_JPY_lowMid'] = OLSResults.load('../models/Linreg_low.h5')

pricediff = True
instrument = 'USD_JPY'

if pricediff:
    Actuals = ['Diff {}_closeMid'.format(instrument),
               'Diff {}_highMid'.format(instrument),
               'Diff {}_lowMid'.format(instrument)]

    Responses = ['future diff close',
                 'future diff high',
                 'future diff low']
else:
    Actuals = ['{}_closeMid'.format(instrument),
               '{}_highMid'.format(instrument),
               '{}_lowMid'.format(instrument)]

    Responses = ['future close',
                 'future high',
                 'future low']

candleformat = 'midpoint' # ['midpoint', 'bidask']
instrument = 'USD_JPY'
instruments = ['USD_JPY',
               'USB02Y_USD',
               'USB05Y_USD',
               'USB10Y_USD',
               'USB30Y_USD',
               'UK100_GBP',
               'UK10YB_GBP',
               'JP225_USD',
               'HK33_HKD',
               'EU50_EUR',
               'DE30_EUR',
               'DE10YB_EUR',
               'WTICO_USD',
               'US30_USD',
               'SPX500_USD']

granularity = 'H1'
start = '2018-11-25'
end = str(dt.now())
freq = 'D'
trading = True

time.sleep(30) #sleep for one second.

fx = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

instrument = 'USD_JPY'
pricediff = True
log = True
min_window = None
candleformat = 'midpoint' # ['midpoint', 'bidask']
trading = True

afx = adjust_lags(fx,
                  min_window=min_window,
                  instrument=instrument,
                  pricediff=pricediff,
                  candleformat=candleformat,
                  log=log,
                  trading=trading)

afx['intercept'] = 1

prices = [i.replace('Diff ', '') for i in Actuals]

prices.append('{}_date'.format(instrument))

fxp = fx[prices]
fxp['{}_date'.format(instrument)] = fxp['{}_date'.format(instrument)].astype(str)
fxp['{}_date'.format(instrument)] = fxp['{}_date'.format(instrument)].str[:13]

fxp = fxp.drop(0)

for i in Actuals:
    df = afx[variables[i]]
    x = df.values
    x = sm.add_constant(x, prepend=True, has_constant='skip')
    imod = 'Linreg_' + i
    mod = models[imod]
    if pricediff:
        act = i.replace('Diff ', '')
    fxp['Future ' + act] = np.exp(mod.predict(x))*fxp[act]

pricepreds = fxp.iloc[-2]

print(fxp[-10:])


# # Clasificación

# In[5]:


variables = {'Diff USD_JPY_closeMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff High-Low',
  'intercept'],
 'Diff USD_JPY_highMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff High-Low',
  'intercept'],
 'Diff USD_JPY_lowMid': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff High-Low',
  'intercept']}

from statsmodels.regression.linear_model import OLSResults
models = {}
models['Logreg_Diff USD_JPY_closeMid'] = OLSResults.load('../models/Logreg_close.h5')
models['Logreg_Diff USD_JPY_highMid'] = OLSResults.load('../models/Logreg_high.h5')
models['Logreg_Diff USD_JPY_lowMid'] = OLSResults.load('../models/Logreg_low.h5')

pricediff = True
instrument = 'USD_JPY'

if pricediff:
    Actuals = ['Diff {}_closeMid'.format(instrument),
               'Diff {}_highMid'.format(instrument),
               'Diff {}_lowMid'.format(instrument)]

    Responses = ['future diff close',
                 'future diff high',
                 'future diff low']
else:
    Actuals = ['{}_closeMid'.format(instrument),
               '{}_highMid'.format(instrument),
               '{}_lowMid'.format(instrument)]

    Responses = ['future close',
                 'future high',
                 'future low']

candleformat = 'midpoint' # ['midpoint', 'bidask']
instrument = 'USD_JPY'
instruments = ['USD_JPY',
               'USB02Y_USD',
               'USB05Y_USD',
               'USB10Y_USD',
               'USB30Y_USD',
               'UK100_GBP',
               'UK10YB_GBP',
               'JP225_USD',
               'HK33_HKD',
               'EU50_EUR',
               'DE30_EUR',
               'DE10YB_EUR',
               'WTICO_USD',
               'US30_USD',
               'SPX500_USD']

granularity = 'H1'
start = '2018-11-25'
end = str(dt.now())
freq = 'D'
trading = True

fx = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

instrument = 'USD_JPY'
pricediff = True
log = True
min_window = None
candleformat = 'midpoint' # ['midpoint', 'bidask']
trading = True

afx = adjust_lags(fx,
                  min_window=min_window,
                  instrument=instrument,
                  pricediff=pricediff,
                  candleformat=candleformat,
                  log=log,
                  trading=trading)

afx['intercept'] = 1

prices = [i.replace('Diff ', '') for i in Actuals]

prices.append('{}_date'.format(instrument))

fxc = fx[prices]
fxc['{}_date'.format(instrument)] = fxc['{}_date'.format(instrument)].astype(str)
fxc['{}_date'.format(instrument)] = fxc['{}_date'.format(instrument)].str[:13]

fxc = fxc.drop(0)

for i in Actuals:
    df = afx[variables[i]]
    x = df.values
    x = sm.add_constant(x, prepend=True, has_constant='skip')
    imod = 'Logreg_' + i
    mod = models[imod]
    if pricediff:
        act = i.replace('Diff ', '')
    fxc['Future ' + act] = mod.predict(x)

classpreds = fxc.iloc[-2]

print(fxc[-10:])


# In[6]:


fxc


# In[7]:


pricepreds


# In[8]:


classpreds


# In[9]:


actual_high_price = pricepreds['USD_JPY_highMid']
actual_low_price = pricepreds['USD_JPY_lowMid']
future_high_price = pricepreds['Future USD_JPY_highMid']
future_low_price = pricepreds['Future USD_JPY_lowMid']
future_high_class = classpreds['Future USD_JPY_highMid']
future_low_class = 1 - classpreds['Future USD_JPY_lowMid']


# In[10]:


operation = ''
take_profit_buy = ''
take_profit_sell = ''
stop_loss_buy = ''
stop_loss_sell = ''

# Sube de high, No baja de low
if future_high_class > 0.5 and future_low_class < 0.5:
    operation = 'Buy'
    stop_loss_buy = actual_low_price - 0.01
    if future_high_price > actual_high_price:
        take_profit_buy = future_high_price
    else:
        take_profit_buy = actual_high_price

# Baja de low, No sube de high
if future_high_class < 0.5 and future_low_class > 0.5:
    operation = 'Sell'
    stop_loss_sell = actual_high_price + 0.01
    if future_low_price < actual_low_price:
        take_profit_sell = future_low_price
    else:
        take_profit_sell = actual_low_price

# Pasa ambos
if future_high_class > 0.5 and future_low_class > 0.5:
    operation = 'Buy & Sell'
    if future_high_price > actual_high_price:
        take_profit_buy = future_high_price
    else:
        take_profit_buy = actual_high_price

    if future_low_price < actual_low_price:
        take_profit_sell = future_low_price
    else:
        take_profit_sell = actual_low_price


# In[11]:


from datetime import datetime as dt


# In[12]:


print(str(dt.now())[:19])

print('\nOperation: ' + operation)

print('\nBuy Take Profit: ' + str(take_profit_buy))
print('Buy Stop Loss: ' + str(stop_loss_buy))

print('\nSell Take Profit: ' + str(take_profit_sell))
print('Sell Stop Loss: ' + str(stop_loss_sell))


# # Multiclass

# In[13]:


variables = {'Diff USD_JPY_closeMid1': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_closeMid2': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_closeMid3': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_closeMid4': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_closeMid5': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_closeMid6': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid1': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid2': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid3': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid4': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid5': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_highMid6': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid1': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid2': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid3': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid4': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid5': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept'],
 'Diff USD_JPY_lowMid6': ['USD_JPY_volume',
  'USB02Y_USD_volume',
  'USB05Y_USD_volume',
  'USB10Y_USD_volume',
  'USB30Y_USD_volume',
  'UK100_GBP_volume',
  'UK10YB_GBP_volume',
  'JP225_USD_volume',
  'HK33_HKD_volume',
  'EU50_EUR_volume',
  'DE30_EUR_volume',
  'DE10YB_EUR_volume',
  'WTICO_USD_volume',
  'US30_USD_volume',
  'SPX500_USD_volume',
  'Diff USD_JPY_openMid',
  'Diff USD_JPY_volume',
  'Diff USB02Y_USD_closeMid',
  'Diff USB02Y_USD_highMid',
  'Diff USB02Y_USD_lowMid',
  'Diff USB02Y_USD_openMid',
  'Diff USB02Y_USD_volume',
  'Diff USB05Y_USD_closeMid',
  'Diff USB05Y_USD_highMid',
  'Diff USB05Y_USD_lowMid',
  'Diff USB05Y_USD_openMid',
  'Diff USB05Y_USD_volume',
  'Diff USB10Y_USD_closeMid',
  'Diff USB10Y_USD_highMid',
  'Diff USB10Y_USD_lowMid',
  'Diff USB10Y_USD_openMid',
  'Diff USB10Y_USD_volume',
  'Diff USB30Y_USD_closeMid',
  'Diff USB30Y_USD_highMid',
  'Diff USB30Y_USD_lowMid',
  'Diff USB30Y_USD_openMid',
  'Diff USB30Y_USD_volume',
  'Diff UK100_GBP_closeMid',
  'Diff UK100_GBP_highMid',
  'Diff UK100_GBP_lowMid',
  'Diff UK100_GBP_openMid',
  'Diff UK100_GBP_volume',
  'Diff UK10YB_GBP_closeMid',
  'Diff UK10YB_GBP_highMid',
  'Diff UK10YB_GBP_lowMid',
  'Diff UK10YB_GBP_openMid',
  'Diff UK10YB_GBP_volume',
  'Diff JP225_USD_closeMid',
  'Diff JP225_USD_highMid',
  'Diff JP225_USD_lowMid',
  'Diff JP225_USD_openMid',
  'Diff JP225_USD_volume',
  'Diff HK33_HKD_closeMid',
  'Diff HK33_HKD_highMid',
  'Diff HK33_HKD_lowMid',
  'Diff HK33_HKD_openMid',
  'Diff HK33_HKD_volume',
  'Diff EU50_EUR_closeMid',
  'Diff EU50_EUR_highMid',
  'Diff EU50_EUR_lowMid',
  'Diff EU50_EUR_openMid',
  'Diff EU50_EUR_volume',
  'Diff DE30_EUR_closeMid',
  'Diff DE30_EUR_highMid',
  'Diff DE30_EUR_lowMid',
  'Diff DE30_EUR_openMid',
  'Diff DE30_EUR_volume',
  'Diff DE10YB_EUR_closeMid',
  'Diff DE10YB_EUR_highMid',
  'Diff DE10YB_EUR_lowMid',
  'Diff DE10YB_EUR_openMid',
  'Diff DE10YB_EUR_volume',
  'Diff WTICO_USD_closeMid',
  'Diff WTICO_USD_highMid',
  'Diff WTICO_USD_lowMid',
  'Diff WTICO_USD_openMid',
  'Diff WTICO_USD_volume',
  'Diff US30_USD_closeMid',
  'Diff US30_USD_highMid',
  'Diff US30_USD_lowMid',
  'Diff US30_USD_openMid',
  'Diff US30_USD_volume',
  'Diff SPX500_USD_closeMid',
  'Diff SPX500_USD_highMid',
  'Diff SPX500_USD_lowMid',
  'Diff SPX500_USD_openMid',
  'Diff SPX500_USD_volume',
  'Diff High-Close',
  'Diff Close-Low',
  'intercept']}

from statsmodels.regression.linear_model import OLSResults
models = {}
models['Logreg_Diff USD_JPY_closeMid1'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid1.h5')
models['Logreg_Diff USD_JPY_highMid1'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid1.h5')
models['Logreg_Diff USD_JPY_lowMid1'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid1.h5')

models['Logreg_Diff USD_JPY_closeMid2'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid2.h5')
models['Logreg_Diff USD_JPY_highMid2'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid2.h5')
models['Logreg_Diff USD_JPY_lowMid2'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid2.h5')

models['Logreg_Diff USD_JPY_closeMid3'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid3.h5')
models['Logreg_Diff USD_JPY_highMid3'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid3.h5')
models['Logreg_Diff USD_JPY_lowMid3'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid3.h5')

models['Logreg_Diff USD_JPY_closeMid4'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid4.h5')
models['Logreg_Diff USD_JPY_highMid4'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid4.h5')
models['Logreg_Diff USD_JPY_lowMid4'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid4.h5')

models['Logreg_Diff USD_JPY_closeMid5'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid5.h5')
models['Logreg_Diff USD_JPY_highMid5'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid5.h5')
models['Logreg_Diff USD_JPY_lowMid5'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid5.h5')

models['Logreg_Diff USD_JPY_closeMid6'] = OLSResults.load('../models/Logreg_Diff USD_JPY_closeMid6.h5')
models['Logreg_Diff USD_JPY_highMid6'] = OLSResults.load('../models/Logreg_Diff USD_JPY_highMid6.h5')
models['Logreg_Diff USD_JPY_lowMid6'] = OLSResults.load('../models/Logreg_Diff USD_JPY_lowMid6.h5')


pricediff = True
instrument = 'USD_JPY'

if pricediff:
    Actuals = ['Diff {}_closeMid'.format(instrument),
               'Diff {}_highMid'.format(instrument),
               'Diff {}_lowMid'.format(instrument)]

    Responses = ['future diff close',
                 'future diff high',
                 'future diff low']
else:
    Actuals = ['{}_closeMid'.format(instrument),
               '{}_highMid'.format(instrument),
               '{}_lowMid'.format(instrument)]

    Responses = ['future close',
                 'future high',
                 'future low']

candleformat = 'midpoint' # ['midpoint', 'bidask']
instrument = 'USD_JPY'
instruments = ['USD_JPY',
               'USB02Y_USD',
               'USB05Y_USD',
               'USB10Y_USD',
               'USB30Y_USD',
               'UK100_GBP',
               'UK10YB_GBP',
               'JP225_USD',
               'HK33_HKD',
               'EU50_EUR',
               'DE30_EUR',
               'DE10YB_EUR',
               'WTICO_USD',
               'US30_USD',
               'SPX500_USD']

granularity = 'H1'
start = '2018-11-25'
end = str(dt.now())
freq = 'D'
trading = True

fx = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

instrument = 'USD_JPY'
pricediff = True
log = True
min_window = None
candleformat = 'midpoint' # ['midpoint', 'bidask']
trading = True

afx = adjust_lags(fx,
                  min_window=min_window,
                  instrument=instrument,
                  pricediff=pricediff,
                  candleformat=candleformat,
                  log=log,
                  trading=trading)

afx['intercept'] = 1

prices = [i.replace('Diff ', '') for i in Actuals]

prices.append('{}_date'.format(instrument))

fxcm = fx[prices]
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].astype(str)
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].str[:13]

fxcm = fxcm.drop(0)


# In[14]:


fxcm[-10:]


# In[15]:


for i in Actuals:
    for k in [1,2,3,4,5,6]:
        df = afx[variables[i + str(k)]]
        x = df.values
        x = sm.add_constant(x, prepend=True, has_constant='skip')
        imod = 'Logreg_' + i + str(k)
        mod = models[imod]
        if pricediff:
            act = i.replace('Diff ', '')
        fxcm['Future ' + act + str(k)] = mod.predict(x)

classpredsm = fxcm.iloc[-2]


# In[16]:


classpredsm = pd.DataFrame(fxcm.iloc[-2])
classpredsm.columns = ['Prices']


# In[17]:


classpreds['Future USD_JPY_lowMid'] = 1 - classpreds['Future USD_JPY_lowMid']




classpreds




classpredsm = classpredsm.drop('USD_JPY_date')


# In[20]:


new_preds = classpredsm.iloc[:3]


# In[21]:


new_preds.columns = ['Prices']


# In[22]:


for i in [1,2,3,4,5,6]:
    l = [j for j in classpredsm.index if str(i) in j]
    new_preds['p(' + str(i) + ')'] = classpredsm.loc[l]['Prices'].values
    new_preds[str(i/100) + '%'] = 0
    new_preds[str(i/100) + '%'].iloc[:2] = new_preds['Prices'] * (1+i/10000)
    new_preds[str(i/100) + '%'].iloc[-1:] = new_preds['Prices'] * (1-i/10000)


# In[23]:


actual_high_price = classpreds['USD_JPY_highMid']
actual_low_price = classpreds['USD_JPY_lowMid']

close_up = classpreds['Future USD_JPY_closeMid']
high_up = classpreds['Future USD_JPY_highMid']
low_down = classpreds['Future USD_JPY_lowMid']


# In[24]:


new_preds.insert(1,'p(0)',[close_up, high_up, low_down])


# In[25]:


new_preds = new_preds.round(3)


# In[26]:


new_preds


# In[27]:


probas = [i for i in new_preds.columns if 'p(' in i]
prices = [i for i in new_preds.columns if '%' in i or 'P' in i]
indx = 'Buy'

op_buy = pd.DataFrame({'Take Profit': np.zeros(7)},
                  index=[indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])

op_buy['Take Profit'] = new_preds[prices].iloc[1].values
op_buy['Proba TP'] = new_preds[probas].iloc[1].values

op_buy['Stop Loss'] = new_preds[prices].iloc[2].values
op_buy['Proba SL'] = new_preds[probas].iloc[2].values

indx = 'Sell'

op_sell = pd.DataFrame({'Take Profit': np.zeros(7)},
                  index=[indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])

op_sell['Take Profit'] = new_preds[prices].iloc[2].values
op_sell['Proba TP'] = new_preds[probas].iloc[2].values

op_sell['Stop Loss'] = new_preds[prices].iloc[1].values
op_sell['Proba SL'] = new_preds[probas].iloc[1].values




print(op_buy)
print(op_sell)

def send_email(subject,fromaddr, toaddr, password,body_text):
    """
    Manda email de tu correo a tu correo

    Args:
        subject (str): Asunto del correo
        body_test (str): Cuerpo del correo
    """
    html_template = open("./email/email_template.html", 'r')
    html_template = html_template.read()

    # datetime object with timezone awareness:
    datetime.datetime.now(tz=pytz.utc)

    # seconds from epoch:
    datetime.datetime.now(tz=pytz.utc).timestamp()

    # ms from epoch:
    hora_now = int(datetime.datetime.now(tz=pytz.utc).timestamp() * 1000)
    hora_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    #mail = os.environ['EMAIL']
    #password = os.environ['EMAIL_PASSWORD']
    msg = MIMEMultipart()
    msg.preamble = f'Predicciones de la hora {hora_now}'
    msg['From'] = fromaddr
    msg['To'] = COMMASPACE.join(toaddr)
    msg['Subject'] = subject+' '+str(hora_now)

    soup = BeautifulSoup(html_template, features="lxml")
    find_buy = soup.find("table", {"id": "buy_table"})
    br = soup.new_tag('br')

    for i, table in enumerate(soup.select('table.dataframe')):
        print(f'i: {i}')
        print(f'body_text[{i}]: {body_text[i]}')
        table.replace_with(BeautifulSoup(body_text[i].to_html(), "html.parser"))


    msg.attach(MIMEText(soup, 'html'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


send_email('Predicciones de USDJPY', 'prophetsfai@gmail.com',['deds15@gmail.com', 'franserr93@gmail.com', 'ricardomd16@gmail.com'],"prophetsai18", [op_buy, op_sell])
