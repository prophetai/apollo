#!/usr/bin/env python
# coding: utf-8

from data import Data
from processData.processing import setup_data, get_indicators
from loadAssets import Assets
from getData.extract import get_forex
from keras.models import load_model
from sklearn import preprocessing
import pickle
import argparse
import pytz
import os
import sys
import getopt
import logging
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLSResults

from trade.logic import Decide
from trade.order import Order
from trade.trade import Trade
from trade.openTrades import openTrades

from send_predictions.email_send import send_email, create_html, from_html_to_jpg, make_image
from send_predictions.telegram_send import telegram_bot


# utilities propias
from check_market import market_open
# librerías para manejo de tiempo
from datetime import timedelta
from datetime import datetime as dt
from datetime import date
import time

import warnings
warnings.filterwarnings("ignore")
sys.path.append('./src/assets/')
sys.path.append('./src')

pd.options.display.max_columns = None

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='/tmp/log_test.txt'
)

# constantes para extraer datos
logging.info('\n\n************* Inicio **************')

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
               'SPX500_USD',
               'NL25_EUR',
               'FR40_EUR',
               'AU200_AUD',
               'US2000_USD',
               'XAU_USD',
               'AUD_USD',
               'GBP_USD',
               'USD_CAD',
               'USD_CHF',
               'EUR_USD']

instrument = 'USD_JPY',
model_version = 'models'

# Carga de assets
logging.info('************* Carga de modelos **************')
assets = Assets(model_version, instrument)
model_files = assets.load_models()

# Descarga de datos
logging.info('************* Descargando datos **************')
# time.sleep(3)
data = Data(instrument=instrument,
            ins_variables=instruments,
            granularity='H1',
            start=str(dt.now() + timedelta(days=-2))[:10],
            end=str(dt.now())[:10],
            freq='D',
            candleformat='bidask',
            trading=True)

Xh, Xl = data.get_data(model_version)

###

models = {f'Xh_gbH_{i}': file_model.predict_proba(Xh)[:, 1] for i, file_model in enumerate(model_files[:len(model_files)/2])}
models.update({f'Xl_gbl_{i}': file_model.predict_proba(Xl)[:,1] for i, file_model in enumerate(model_files[len(model_files)/2:])})


# predicciones
logging.info('************* Haciendo Predicciones **************')

preds_buy = {
    'Xh_gbH_6': models['Xh_gbH_6'][-2],
    'Xh_gbH_5': models['Xh_gbH_5'][-2],
    'Xh_gbH_4': models['Xh_gbH_4'][-2],
    'Xh_gbH_3': models['Xh_gbH_3'][-2],
    'Xh_gbH_2': models['Xh_gbH_2'][-2],
    'Xh_gbH_1': models['Xh_gbH_1'][-2],
    'Xh_gbH0': models['Xh_gbH0'][-2],
    'Xh_gbH1': models['Xh_gbH1'][-2],
    'Xh_gbH2': models['Xh_gbH2'][-2],
    'Xh_gbH3': models['Xh_gbH3'][-2],
    'Xh_gbH4': models['Xh_gbH4'][-2],
    'Xh_gbH5': models['Xh_gbH5'][-2],
    'Xh_gbH6': models['Xh_gbH6'][-2]
}

preds_sell = {
    'Xl_gbl_6': models['Xl_gbl_6'][-2],
    'Xl_gbl_5': models['Xl_gbl_5'][-2],
    'Xl_gbl_4': models['Xl_gbl_4'][-2],
    'Xl_gbl_3': models['Xl_gbl_3'][-2],
    'Xl_gbl_2': models['Xl_gbl_2'][-2],
    'Xl_gbl_1': models['Xl_gbl_1'][-2],
    'Xl_gbl0': models['Xl_gbl0'][-2],
    'Xl_gbl1': models['Xl_gbl1'][-2],
    'Xl_gbl2': models['Xl_gbl2'][-2],
    'Xl_gbl3': models['Xl_gbl3'][-2],
    'Xl_gbl4': models['Xl_gbl4'][-2],
    'Xl_gbl5': models['Xl_gbl5'][-2],
    'Xl_gbl6': models['Xl_gbl6'][-2]
}

pricediff = True

Actuals = ['HHLL_LogDiff {}_highAsk'.format(instrument),
           'HHLL_LogDiff {}_highBid'.format(instrument),
           'HHLL_LogDiff {}_lowAsk'.format(instrument),
           'HHLL_LogDiff {}_lowBid'.format(instrument)]

Responses = ['future diff high',
             'future diff low']

prices = [i.replace('HHLL_LogDiff ', '') for i in Actuals]
prices.append('{}_date'.format(instrument))

fxcm = gf[prices]
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(
    instrument)].astype(str)
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(
    instrument)].str[:13]

fxcm = fxcm.drop(0)

variableshhll = pd.read_csv(
    './src/assets/variables/variablesHigh.csv', index_col=0)

current_open_ask = gf['USD_JPY_openAsk'].iloc[-1].round(3)
current_open_bid = gf['USD_JPY_openBid'].iloc[-1].round(3)

# aquí se toma el precio más caro a la venta de la hora previa (investigar por qué aquí es el último dato)
previous_high_ask = gf['USD_JPY_highAsk'].iloc[-2].round(3)
take_profit_buy = [((1+i/(10000.00))*previous_high_ask).round(3)
                   for i in range(-6, 7)]

# aquí se toma el precio más barato a la compra de la hora previa (investigar por qué aquí es el penúltimo dato)
previous_low_bid = gf['USD_JPY_lowBid'].iloc[-2].round(3)
take_profit_sell = [((1-i/(10000.00))*previous_low_bid).round(3)
                    for i in range(-6, 7)]

op_buy = pd.DataFrame({'Open': current_open_ask,
                       'Probability': list(preds_buy.values()),
                       'Take Profit': take_profit_buy},
                      index=[f'Buy{i}' for i in range(-6, 7)])

op_sell = pd.DataFrame({'Open': current_open_bid,
                        'Probability': list(preds_sell.values()),
                        'Take Profit': take_profit_sell},
                       index=[f'Sell{i}' for i in range(-6, 7)])

op_buy.loc[op_buy['Open'] > op_buy['Take Profit'],
           ['Take Profit', 'Probability']] = np.nan
op_sell.loc[op_sell['Open'] < op_sell['Take Profit'],
            ['Take Profit', 'Probability']] = np.nan


new = [0, 2, 1]
op_buy = op_buy[op_buy.columns[new]]
op_sell = op_sell[op_sell.columns[new]]


def main(argv):
    """
    Main
    """
    TOKEN = os.environ['telegram_token']
    CHAT_ID = os.environ['telegram_chat_id']
    initial_pip = float(os.environ['initial_pip'])
    html_template_path = "./src/assets/email/email_template.html"

    tz_MX = pytz.timezone('America/Mexico_City')
    datetime_MX = dt.now(tz_MX)

    hora_now = f'{datetime_MX.strftime("%H:%M:%S")}'

    parser = argparse.ArgumentParser(description='Apollo V 0.1 Beta')

    parser.add_argument('-o', '--order', action='store_true',
                        help='Determine if you want to make an order')
    parser.add_argument('-t', '--time', action='store_true',
                        help='Make order only if market is open')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug mode on')
    parser.add_argument('-p', '--practice', action='store_true',
                        help='Use practice account too')
    parser.add_argument()

    args = parser.parse_args()
    make_order = args.order or False
    market_sensitive = args.time or False
    debug_mode = args.debug or False
    practice_on = args.practice or False

    logging.info(f'\nMarket sensitive: {market_sensitive}')
    if market_sensitive and not market_open():
        logging.info('Market Closed')
        return

# Hacer decisón para la posición
    decision = Decide(op_buy, op_sell, 100000, direction=0,
                      pips=initial_pip, take_profit=0)
    decision.get_all_pips()
    units = decision.pips * decision.direction * 1000

    # máximo de unidades en riesgo al mismo tiempo
    pip_limit = float(os.environ['pip_limit'])
    open_trades = openTrades()
    current_pips = open_trades.number_trades()

    print(f'Current units: {current_pips}')
    print(f'Max units: {pip_limit}')
    print(f'Units: {units}')

    # si queremos hacer una operación (units puede ser positivo o negativo)
    if units != 0:
        if current_pips < pip_limit:  # vemos si aún podemos hacer operaciones
            # escogemos lo que podamos operar sin pasarnos del límite.
            # el mínimo entre la unidades solicitadas o las disponibles
            units = min(abs(units), pip_limit - current_pips) * \
                decision.direction
            if units == 0.0:  # si encontramos que ya no hay
                decision.decision += '\n*Units limit exceeded. Order not placed.'
        else:  # si ya hemos excedido operaciones
            units = 0.0
            decision.decision += '\n*Units limit exceeded. Order not placed.'

    inv_instrument = 'USD_JPY'
    take_profit = decision.take_profit
    op_buy_new = decision.data_buy
    op_sell_new = decision.data_sell

    print(f'inv_instrument: {inv_instrument}')
    print(f'take_profit: {take_profit}')

    logging.info(f'\n{decision.decision}')
    # Pone orden a precio de mercado
    logging.info(
        f'Units: {units}, inv_instrument: {inv_instrument} , take_profit: {take_profit}\n')

    if make_order and units != 0:
        new_trade = Trade(inv_instrument, units, take_profit=take_profit)
        new_order = Order(new_trade)
        new_order.make_market_order()

    if practice_on:
        # token de autenticación
        os.environ['token'] = os.environ['token_demo']
        # URL de broker
        os.environ['trading_url'] = os.environ['trading_url_demo']
        open_trades = openTrades()
        current_pips = open_trades.number_trades()
        print(f'trading url: {os.environ["trading_url"]}')
        if units != 0:  # if we want to make a trade
            print(f'inv_instrument: {inv_instrument}')
            print(f'take_profit: {take_profit}')
            new_order = Order(new_trade)
            new_order.make_market_order()

        print(f'checking open trades({current_pips} pips)')
        if current_pips > 0:  # if there are any open positions
            open_trades.get_trades_data()
            # calculate the stop loss for open trades
            for trade in open_trades.trades:
                from dateutil import parser
                duration = dt.now().astimezone(pytz.utc) - parser.parse(trade.openTime)
                minutes = duration.total_seconds()/60
                print(f'ID:{trade.i_d}')
                print(f'minutes passed for trade {trade.i_d}: {minutes}')
                if minutes > 59 and minutes < 120:  # just for trades with more than 50 minutes old
                    trade.get_stop_loss()
                    new_order = Order(trade)
                    new_order.set_stop_loss()  # sets it in Oanda

    print(f'\nPrevious High Ask:{previous_high_ask}')
    print(op_buy_new)
    print(f'\nPrevious Low Bid: {previous_low_bid}')
    print(op_sell_new)

    # send telegram
    if not debug_mode:
        _, html_path = create_html(
            [op_buy, op_sell, previous_high_ask, previous_low_bid], html_template_path)
        _, image_name = from_html_to_jpg(html_path)
        logging.info('Se mandan predicciones a Telegram')
        bot = telegram_bot(TOKEN)
        if not make_order:
            bot.send_message(CHAT_ID, f"TEST!!!!!")
        bot.send_message(CHAT_ID, f"Predictions for the hour: {hora_now}")
        bot.send_photo(CHAT_ID, image_name)
        bot.send_message(
            CHAT_ID, f"Best course of action: {decision.decision}")


if __name__ == '__main__':
    # load settings
    with open("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
