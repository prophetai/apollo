#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import argparse
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
def save_instrument_history(conn_data, instrument="USD_JPY"):

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

def main(argv):
    """
    Main
    """
    parser = argparse.ArgumentParser(description='Homer V 0.01 Beta')

    parser.add_argument('-u', '--user',
                        help='Database user')
    parser.add_argument('-p', '--password',
                        help='Database password')
    parser.add_argument('-h', '--host',
                        help='database host')
    parser.add_argument('-n', '--name',
                        help='database name')

    args = parser.parse_args()
    user = args.user
    passwd = args.password
    host = args.host
    name = args.name

    conn_data = {
        'db_user': user,
        'db_pwd': passwd,
        'db_host': host,
        'db_name': name
    }

    save_instrument_history(conn_data, instrument="USD_JPY")
