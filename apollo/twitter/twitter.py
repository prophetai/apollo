#-*- coding: utf-8 -*-
"""
Funciones para obtener los datos de twitter en la base de datos
"""
import sys, getopt
import json
import pandas as pd
import logging
import twint
import datetime
sys.path.insert(0, '..')
from utils.extract import db_connection, download_data
import google.cloud.logging
from textblob import TextBlob

# Instancia un cliente para el logger
#client = google.cloud.logging.Client()

# Connects the logger to the root logging handler; by default this captures
# all logs at INFO level and higher
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)

#client.setup_logging()

def search_tweets(cuenta, debug=False):
    """
    Obtiene todos los tweets de una cuenta

    Args:
        cuenta(str): nombre de la cuenta a buscar
    Response:
        df(DataFrame): DataFrame con los resultados
    """

    c = twint.Config()
    c.Username = cuenta.replace('@','')
    c.Pandas = True

    if debug:
        c.Limit = 20
        #cuando esté, hacer el cambio para que no se impriman los tweets

    try:
        twint.run.Search(c)
    except asyncio.TimeoutError as error:
        logging.error('Error en la búsqueda de tweets: %s' % (error))

    df = twint.storage.panda.Tweets_df
    Pandas_clean = True
    return df

def get_sentiment(tweet):
    text = TextBlob(tweet)
    polarity = text.polarity
    subjec = text.subjectivity

    return text, polarity, subjec

def load_tweets(DF, creds, debug=False):
    """
    Carga los tweets desde un dataframe a una base de datos

    Args:
        df(Dataframe): DataFrame con datos a subir a la base de datos
        creds(dict): Diccionario con las credenciales de la base de datos
    """
    logging.info('*** Cargando tweets ***')
    df = DF.copy()

    new_order = ['id', 'user_id', 'date', 'timezone', 'location', 'username', 'tweet', 'hashtags', 'link', 'retweet', 'user_rt', 'mentions']
    df = df[new_order]
    df['hashtags'].replace('[','',inplace=True)
    df['hashtags'].replace(']','',inplace=True)

    lista_tweets = df.values.tolist()

    data_ready = ''
    for i, tweet in enumerate(lista_tweets, start=0):
        tweet_str = []
        cuenta = ''
        for j, element in enumerate(tweet):
            if j == 5:
                cuenta = element
            if j == 6:
                sentiment = get_sentiment(element)
            element = str(element).replace("'", '')
            transform = "" + str(element).replace("['", '[').replace("']",']')
            tweet_str.append(transform)
        tweet_str.append(sentiment[1])
        tweet_str.append(sentiment[2])
        data_ready += "(" + str(tweet_str)[1:-1] + ")"

        if i % 10000 == 0 and data_ready != [] and i > 0:
            try:
                data_ready = data_ready.replace(")(",'), (')
                query = """INSERT INTO tweets (id, user_id , date , timezone , location , username , tweet , hashtags , link , retweet , user_rt , mentions, polarity , subjectivity ) VALUES {} ON CONFLICT (id) DO NOTHING;""".format(data_ready)
                if debug:
                    logging.error('query: {}'.format(query))
                conn = db_connection(creds)
                download_data(conn, query)
                data_ready = ''
                logging.info("Se guardaron tweets ({}-{}) de la cuenta {}".format(i-10000,len(lista_tweets), cuenta))
            except Exception as error:
                logging.error("Error al tratar de insertar: %s" % (error))
        elif i == len(lista_tweets)-1 and data_ready != []:
            try:
                data_ready = data_ready.replace(")(",'), (')
                query = """INSERT INTO tweets (id, user_id , date , timezone , location , username , tweet , hashtags , link , retweet , user_rt , mentions, polarity , subjectivity ) VALUES {} ON CONFLICT (id) DO NOTHING;""".format(data_ready)
                if debug:
                    logging.error('query: {}'.format(query))
                conn = db_connection(creds)
                download_data(conn, query)
                logging.info("Se guardaron los últimos tweets ({}-{})".format(i - len(lista_tweets) + 1,len(lista_tweets)))
            except Exception as error:
                logging.error("Error al tratar de insertar: %s" % (error))

    logging.info('Se terminan de guardar todos los tweets de {}'.format(cuenta))

def main(argv):
    """
    Corre la actualización de una lista de tweets
    """
    debug = False
    logging.info('Iniciando extracción de tweets')
    try:
      opts, args = getopt.getopt(argv,"ha:c:d:",["accounts=","creds=","debug="])
    except getopt.GetoptError:
        print('twitter.py -a <ruta de archivo de cuentas> -c <ruta a creds>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('twitter.py -a <ruta de archivo de cuentas> -c <ruta a creds>')
            sys.exit()
        elif opt in ("-a", "--accounts"):
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
        lista_cuentas = pd.read_csv(inputfile)['Cuentas']
    except Exception as e:
        logging.error('No se encuentra el archivo de cuentas: {} {}'.format(inputfile,e))
        sys.exit(2)

    for cuenta in lista_cuentas:
        df = search_tweets(cuenta, debug=debug)
        load_tweets(df, creds, debug=debug)


if __name__ == "__main__":
    version = ".".join(str(v) for v in sys.version_info[:2])
    if float(version) < 3.6:
        print("[-] TWINT requires Python version 3.6+.")
        sys.exit(0)

    main(sys.argv[1:])
