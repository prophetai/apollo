#-*- coding: utf-8 -*-
import os, sys, getopt
from os.path import dirname, join, abspath
import logging
from pandas_datareader import data
import pandas as pd
import datetime
import json
import numpy as np
import google.cloud.logging
sys.path.insert(0, '..')
from utils.extract import db_connection, download_data

now = datetime.datetime.now()
# Instancia un cliente para el logger
client = google.cloud.logging.Client()

# Connects the logger to the root logging handler; by default this captures
# all logs at INFO level and higher
client.setup_logging()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

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

def create_new_stock_table(table_name, creds):
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
def get_stock_data(lista_tickers, creds, previous_date=False):
    """
    Obtiene los datos de precios de acciones y los guarda en la base de datos

    Args:
        lista_tickers(list): lista de strings con los nombres de los tickers como vienen en Yahoo Finance
        end_date(string): String que indica la fecha de hasta donde recolectar datos
        creds(dict): diccionario de credenciales de la base de datos
    """
    end_date = '{}-{}-{}'.format(now.year, now.month, now.day)
    for ticker in lista_tickers:
        if previous_date:
            start_date = get_last_date(ticker.replace('.','_').replace('-','_'), creds)
            logging.warning("start_date: %s end_date: %s" % (start_date, end_date))
        else:
            start_date = end_date

        try:
            panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
            table_name = ticker.replace('.','_').replace('-','_')
            create_new_stock_table(table_name, creds)
            update_in_db(panel_data, table_name, creds)

        except Exception as error:
            logging.error("Error al tratar de obtener datos de %s: %s" % (ticker,error))

def main(argv):
    """
    Corre la actualización de una lista de stocks
    """
    debug = False
    logging.info('Iniciando actualización de stocks')
    try:
      opts, args = getopt.getopt(argv,"ht:c:d:",["tickers=","creds=","debug="])
    except getopt.GetoptError:
        print('tickers.py -a <ruta de archivo de tickers> -c <ruta a creds>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('tickers.py -t <ruta de archivo de tickers> -c <ruta a creds>')
            sys.exit()
        elif opt in ("-t", "--tickers"):
            inputfile = arg
        elif opt in ("-c", "--creds"):
            creds_file = arg
        elif opt in ("-d", "--debug"):
            debug = True
    try:
        with open(creds_file, encoding='utf-8') as data_file:
            creds = json.loads(data_file.read())
    except Exception as e:
        logging.error('No se encuentra el archivo de credenciales: {}, {}'.format(creds_file,e))
        sys.exit(2)
    try:
        lista_tickers = pd.read_csv(inputfile)['Ticker']
    except Exception as e:
        logging.error('No se encuentra el archivo de cuentas: {} {}'.format(inputfile,e))
        sys.exit(2)

    get_stock_data(lista_tickers,creds)

if __name__ == "__main__":
    main(sys.argv[1:])
