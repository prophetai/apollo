#-*- coding: utf-8 -*-
import os, sys
from os.path import dirname, join, abspath
import logging
from pandas_datareader import data
import pandas as pd
import datetime
import json
import numpy as np
import oandapy as opy
import google.cloud.logging


from utils.extract import db_connection, download_data
# Instancia un cliente para el logger
client = google.cloud.logging.Client()

# Connects the logger to the root logging handler; by default this captures
# all logs at INFO level and higher
client.setup_logging()

#Inicializamos el api de Oanda
oanda = opy.API(environment='live')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

with open('creds.txt', encoding='utf-8') as data_file:
    creds = json.loads(data_file.read())

now = datetime.datetime.now()



def get_last_date(table_name, creds):
    """
    Trae la fecha del último precio guardado
    """
    try:
        query = "SELECT MAX(date) FROM {}".format(table_name)
        conn = db_connection(creds)
        df = download_data(conn, query)
        latest_date = str(df[0][0])
    except Exception as e:
        logging.error('Error sacando la última fecha de %s: %s' % (table_name, e))
        lates_date = '2010-01-01'

    return latest_date

def create_new_forex_table(table_name, creds):
    """
    Hace una nueva tabla de stocks según el ticker que le pongas

    Args:
        table_name(str): nombre del ticker
        conn(connection object): objeto de connección a la base de datos
    """
    conn = db_connection(creds)
    query = """CREATE TABLE IF NOT EXISTS {} (id SERIAL,
    date date NOT NULL UNIQUE,
    high float,
    low float,
    open float,
    close float,
    volume float,
    adj_close float, PRIMARY KEY (id));""".format(table_name)
    download_data(conn, query)

def update_in_db(df, table_name, creds):
    matrix = np.array(df.to_records().view(type=np.matrix))[0]

    data = []

    for i in range(len(matrix)):
        conv_date = pd.to_datetime(matrix[i][0])
        date = "('" + str(conv_date.year) + "-" + str(conv_date.month) + "-" + str(conv_date.day) + "')::date"
        High = str(matrix[i][1])
        Low = str(matrix[i][2])
        Open = str(matrix[i][3])
        Close = str(matrix[i][4])
        Volume = str(matrix[i][5])
        Adj_Clos = str(matrix[i][6])
        prices = "(" + date + ", " + High + ", " + Low + ", " + Open + ", " + Close + ", " + Volume + "," + Adj_Clos +")"
        data.append(prices)

        print(data)
        data = str(data).replace("[", "(").replace("]", ")").replace('(', '', 1)[:-1].replace('"','')
        table_name = table_name.replace('-','_')
        query = """INSERT INTO {} (date, high, low, open, close, volume, adj_close) VALUES {} ON CONFLICT ON CONSTRAINT {}_date_key DO NOTHING;""".format(table_name.upper(), data, table_name.lower())

        if i % 10000 == 0:
            try:
                conn = db_connection(creds)
                download_data(conn, query)
                data = []
                logging.info("Se guardó: {}".format(table_name))
            except Exception as error:
                logging.error("Error al tratar de insertar %s: %s" % (table_name,error))
        elif i == len(matrix)-1:
            try:
                conn = db_connection(creds)
                download_data(conn, query)
                logging.info("Se guardó: {}".format(table_name))
            except Exception as error:
                logging.error("Error al tratar de insertar %s: %s" % (table_name,error))




# User pandas_reader.data.DataReader to load the desired data. As simple as that.
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
        print(j)
        # Extraemos datos cada 2 días (por simplicidad)
        d1 = start
        d2 = end
        dates = pd.date_range(start=d1, end=d2, freq=freq)
        df = pd.DataFrame()
        print('Descargando:')
        pbar = tqdm(total=len(dates) - 1)

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
                    pbar.update(1)
                except:
                    pass
                
        pbar.close()
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
                       right_on=join_id, how='left',
                       validate='one_to_one')
    return dat
instrument = 'USD_JPY'
get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading=False)
