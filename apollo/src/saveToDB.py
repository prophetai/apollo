import os
import logging
import pandas as pd
import numpy as np
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

def save_instrument_history(instrument="USD_JPY"):

    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
    
    candleformat = 'bidask'
    granularity = 'H1'
    start = '2008-01-01'
    end = str(dt.now())[:12]
    freq = '5D'
    trading = True

    data = get_instrument_history(instrument,
                      granularity,
                      start,
                      end,
                      candleformat,
                      freq,
                      trading=trading)

    data = data[-4:]
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')
    data.to_sql('historical_usdjpy', engine, if_exists="append")

def save_decisions(account, model, instrument, decision, conn_data):
    if decision.direction == 1:
        account_type = "live"
        if 'practice' in os.environ['trading_url_'+ account]:
            account_type = "practice"

        ask = decision.data_buy["Open"][0]
        bid = decision.data_sell["Open"][0]
        prediction_used = np.nan
        probability = float(s.partition("Sell")[0].partition("Probability")[2][2:7])
        stop_loss = np.nan
        take_profit = decision.take_profit
        spread = decision.spread
        time = dt.now()
        pips = decision.pips
        trade = "buy"
        user = conn_data['db_user']
        pwd = conn_data['db_pwd']
        host = conn_data['db_host']
        data_base = conn_data['db_name']
        engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')

        data = pd.DataFrame({"account": account,
                             "account_type": account_type,
                             "ask": ask,
                             "bid": bid,
                             "instrument": instrument,
                             "model": model,
                             "prediction_used": prediction_used,
                             "probability": probability,
                             "stop_loss": stop_loss,
                             "take_profit": take_profit,
                             "time": time,
                             "trade": trade}, index=[dt.now()])
        data = data.reset_index(drop=True)
        data.to_sql('trades', engine, if_exists="append")


    elif decision.direction == -1:
        account_type = "live"
        if 'practice' in os.environ['trading_url_'+ account]:
            account_type = "practice"

        ask = decision.data_buy["Open"][0]
        bid = decision.data_sell["Open"][0]
        prediction_used = np.nan
        probability = float(s.partition("Sell")[2].partition("Probability")[2][2:7])
        stop_loss = np.nan
        take_profit = decision.take_profit
        spread = decision.spread
        time = dt.now()
        pips = decision.pips
        trade = "sell"
        user = conn_data['db_user']
        pwd = conn_data['db_pwd']
        host = conn_data['db_host']
        data_base = conn_data['db_name']
        engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')

        data = pd.DataFrame({"account": account,
                             "account_type": account_type,
                             "ask": ask,
                             "bid": bid,
                             "instrument": instrument,
                             "model": model,
                             "prediction_used": prediction_used,
                             "probability": probability,
                             "stop_loss": stop_loss,
                             "take_profit": take_profit,
                             "time": time,
                             "trade": trade}, index=[dt.now()])
        data = data.reset_index(drop=True)
        data.to_sql('trades', engine, if_exists="append")
