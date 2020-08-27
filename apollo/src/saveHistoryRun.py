#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
import pandas as pd
import argparse
import oandapy as opy
from datetime import datetime as dt
from sqlalchemy import create_engine
from getData.extract import get_forex
import pytz


def save_instrument_history(conn_data, instrument="USD_JPY"):

    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')

    candleformat = 'bidask'
    granularity = 'M15'
    
    data_db = pd.read_sql_table('historical_usdjpy', engine, columns=['id','date'])
    data_db['date'] = pd.to_datetime(data_db['date'] , format = '%Y-%m-%dT%H:%M:%S.%f%z',cache = True)
    max_date_db = data_db.iloc[data_db['id'].idxmax()]['date']
    logging.info(f'\nID:{data_db["id"].idxmax()}\nMax date:{max_date_db}')
    
    start = str(max_date_db)[:10]
    end = str(dt.now())[:10]
    freq = 'D'
    trading = True
    
    logging.info(f'\nFrom: {start}\nTo: {end}\n')

    try:
        data = get_forex(instrument,
                        [instrument],
                        granularity,
                        start,
                        end,
                        candleformat,
                        freq,
                        trading=trading)
                            

        data = data.iloc[:,1:]
        data.columns = [str(column).split('_')[-1] for column in list(data.columns)]
        logging.info(list(data.columns))

        data = data[data['date']>max_date_db]
        logging.info(data)
        data.to_sql('historical_usdjpy', engine, if_exists="append", index=False)
    except Exception as e:
        logging.info(e)

def main(argv):
    """
    Main
    """
    parser = argparse.ArgumentParser(description='Homer V 0.01 Beta')

    parser.add_argument('-U', '--user',
                        help='Database user')
    parser.add_argument('-P', '--password',
                        help='Database password')
    parser.add_argument('-H', '--host',
                        help='database host')
    parser.add_argument('-N', '--name',
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

if __name__ == "__main__":
    main(sys.argv)
