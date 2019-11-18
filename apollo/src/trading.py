#!/usr/bin/env python
# coding: utf-8

import argparse
import pytz
import os
import sys, getopt
import logging
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLSResults

from trade.logic import Decide
from trade.order import Order
from trade.orders_status import Positions

from send_predictions.email_send import send_email, create_html, from_html_to_jpg, make_image
from send_predictions.telegram_send import telegram_bot


#utilities propias
from check_market import market_open
#librerías para manejo de tiempo
from datetime import timedelta
from datetime import datetime as dt
from datetime import date
import time

import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn import preprocessing
from keras.models import load_model
sys.path.append('./src/assets/')
sys.path.append('./src')
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators


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
candleformat = 'bidask'
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
               
granularity = 'H1'
start = str(dt.now() + timedelta(days=-2))[:10]
end = str(dt.now())[:10]
freq = 'D'
trading = True

# variables para High
logging.info('************* Cargando Variables **************')
variablesh = pd.read_csv('./src/assets/variables/variablesHigh.csv')
variablesh = list(variablesh['0'].values)

# variables para Low
variablesl = pd.read_csv('./src/assets/variables/variablesLow.csv')
variablesl = list(variablesl['0'].values)


# carga de modelos
logging.info('************* Cargando Modelos  Gradient Boosting (High)**************')
gbH_6 = pickle.load(open('./src/assets/models/gbHigh-6.h5', 'rb'))
gbH_5 = pickle.load(open('./src/assets/models/gbHigh-5.h5', 'rb'))
gbH_4 = pickle.load(open('./src/assets/models/gbHigh-4.h5', 'rb'))
gbH_3 = pickle.load(open('./src/assets/models/gbHigh-3.h5', 'rb'))
gbH_2 = pickle.load(open('./src/assets/models/gbHigh-2.h5', 'rb'))
gbH_1 = pickle.load(open('./src/assets/models/gbHigh-1.h5', 'rb'))
gbH0 = pickle.load(open('./src/assets/models/gbHigh0.h5', 'rb'))
gbH1 = pickle.load(open('./src/assets/models/gbHigh1.h5', 'rb'))
gbH2 = pickle.load(open('./src/assets/models/gbHigh2.h5', 'rb'))
gbH3 = pickle.load(open('./src/assets/models/gbHigh3.h5', 'rb'))
gbH4 = pickle.load(open('./src/assets/models/gbHigh4.h5', 'rb'))
gbH5 = pickle.load(open('./src/assets/models/gbHigh5.h5', 'rb'))
gbH6 = pickle.load(open('./src/assets/models/gbHigh6.h5', 'rb'))



logging.info('************* Cargando Modelos  Gradient Boosting (Low)**************')
gbL_6 = pickle.load(open('./src/assets/models/gbLow-6.h5', 'rb'))
gbL_5 = pickle.load(open('./src/assets/models/gbLow-5.h5', 'rb'))
gbL_4 = pickle.load(open('./src/assets/models/gbLow-4.h5', 'rb'))
gbL_3 = pickle.load(open('./src/assets/models/gbLow-3.h5', 'rb'))
gbL_2 = pickle.load(open('./src/assets/models/gbLow-2.h5', 'rb'))
gbL_1 = pickle.load(open('./src/assets/models/gbLow-1.h5', 'rb'))
gbL0 = pickle.load(open('./src/assets/models/gbLow0.h5', 'rb'))
gbL1 = pickle.load(open('./src/assets/models/gbLow1.h5', 'rb'))
gbL2 = pickle.load(open('./src/assets/models/gbLow2.h5', 'rb'))
gbL3 = pickle.load(open('./src/assets/models/gbLow3.h5', 'rb'))
gbL4 = pickle.load(open('./src/assets/models/gbLow4.h5', 'rb'))
gbL5 = pickle.load(open('./src/assets/models/gbLow5.h5', 'rb'))
gbL6 = pickle.load(open('./src/assets/models/gbLow6.h5', 'rb'))

# descarga de datos
logging.info('************* Descargando datos **************')
time.sleep(3)
gf = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

sd = setup_data(gf,
                instrument=instrument,
                pricediff=True,
                log=True,
                trading=True)

# indicadores que se añaden al modelo
logging.info('************* Creando Indicadores **************')
processeddf = get_indicators(sd, 
                             instrument, 
                             column='{}_closeBid'.format(instrument), 
                             wind=10, 
                             bidask='Bid') 
processeddf = get_indicators(processeddf, 
                             instrument, 
                             column='{}_closeAsk'.format(instrument), 
                             wind=10, 
                             bidask='Ask')

processeddf = processeddf.fillna(method='ffill')
processeddf = processeddf.fillna(method='bfill')

# carga de escaladores
logging.info('************* Escalando Datos **************')
Xh = processeddf[variablesh]
scaler = pickle.load(open('./src/assets/models/scalerH', 'rb'))
Xh = scaler.transform(Xh)


Xl = processeddf[variablesl]
scaler = pickle.load(open('./src/assets/models/scalerL', 'rb'))
Xl = scaler.transform(Xl)

# predicciones
logging.info('************* Haciendo Predicciones **************')

models = {}
models['Xh_gbH-6'] = gbH_6.predict_proba(Xh)[:,1]
models['Xh_gbH-5'] = gbH_5.predict_proba(Xh)[:,1]
models['Xh_gbH-4'] = gbH_4.predict_proba(Xh)[:,1]
models['Xh_gbH-3'] = gbH_3.predict_proba(Xh)[:,1]
models['Xh_gbH-2'] = gbH_2.predict_proba(Xh)[:,1]
models['Xh_gbH-1'] = gbH_1.predict_proba(Xh)[:,1]
models['Xh_gbH0']= gbH0.predict_proba(Xh)[:,1]
models['Xh_gbH1']= gbH1.predict_proba(Xh)[:,1]
models['Xh_gbH2']= gbH2.predict_proba(Xh)[:,1]
models['Xh_gbH3']= gbH3.predict_proba(Xh)[:,1]
models['Xh_gbH4']= gbH4.predict_proba(Xh)[:,1]
models['Xh_gbH5']= gbH5.predict_proba(Xh)[:,1]
models['Xh_gbH6']= gbH6.predict_proba(Xh)[:,1]

models['Xl_gbl-6'] = gbL_6.predict_proba(Xl)[:,1]
models['Xl_gbl-5'] = gbL_5.predict_proba(Xl)[:,1]
models['Xl_gbl-4'] = gbL_4.predict_proba(Xl)[:,1]
models['Xl_gbl-3'] = gbL_3.predict_proba(Xl)[:,1]
models['Xl_gbl-2'] = gbL_2.predict_proba(Xl)[:,1]
models['Xl_gbl-1'] = gbL_1.predict_proba(Xl)[:,1]
models['Xl_gbl0']= gbL0.predict_proba(Xl)[:,1]
models['Xl_gbl1']= gbL1.predict_proba(Xl)[:,1]
models['Xl_gbl2']= gbL2.predict_proba(Xl)[:,1]
models['Xl_gbl3']= gbL3.predict_proba(Xl)[:,1]
models['Xl_gbl4']= gbL4.predict_proba(Xl)[:,1]
models['Xl_gbl5']= gbL5.predict_proba(Xl)[:,1]
models['Xl_gbl6']= gbL6.predict_proba(Xl)[:,1]

Xh_gbH_6 = gbH_6.predict_proba(Xh)[:,1]
Xh_gbH_5 = gbH_5.predict_proba(Xh)[:,1]
Xh_gbH_4 = gbH_4.predict_proba(Xh)[:,1]
Xh_gbH_3 = gbH_3.predict_proba(Xh)[:,1]
Xh_gbH_2 = gbH_2.predict_proba(Xh)[:,1]
Xh_gbH_1 = gbH_1.predict_proba(Xh)[:,1]
Xh_gbH0 = gbH0.predict_proba(Xh)[:,1]
Xh_gbH1 = gbH1.predict_proba(Xh)[:,1]
Xh_gbH2 = gbH2.predict_proba(Xh)[:,1]
Xh_gbH3 = gbH3.predict_proba(Xh)[:,1]
Xh_gbH4 = gbH4.predict_proba(Xh)[:,1]
Xh_gbH5 = gbH5.predict_proba(Xh)[:,1]
Xh_gbH6 = gbH6.predict_proba(Xh)[:,1]

Xl_gbl_6 = gbL_6.predict_proba(Xl)[:,1]
Xl_gbl_5 = gbL_5.predict_proba(Xl)[:,1]
Xl_gbl_4 = gbL_4.predict_proba(Xl)[:,1]
Xl_gbl_3 = gbL_3.predict_proba(Xl)[:,1]
Xl_gbl_2 = gbL_2.predict_proba(Xl)[:,1]
Xl_gbl_1 = gbL_1.predict_proba(Xl)[:,1]
Xl_gbl0 = gbL0.predict_proba(Xl)[:,1]
Xl_gbl1 = gbL1.predict_proba(Xl)[:,1]
Xl_gbl2 = gbL2.predict_proba(Xl)[:,1]
Xl_gbl3 = gbL3.predict_proba(Xl)[:,1]
Xl_gbl4 = gbL4.predict_proba(Xl)[:,1]
Xl_gbl5 = gbL5.predict_proba(Xl)[:,1]
Xl_gbl6 = gbL6.predict_proba(Xl)[:,1]

preds_buy = {
'Xh_gbH_6':Xh_gbH_6[-2],
'Xh_gbH_5':Xh_gbH_5[-2],
'Xh_gbH_4':Xh_gbH_4[-2],
'Xh_gbH_3':Xh_gbH_3[-2],
'Xh_gbH_2':Xh_gbH_2[-2],
'Xh_gbH_1':Xh_gbH_1[-2],
'Xh_gbH0':Xh_gbH0[-2],
'Xh_gbH1':Xh_gbH1[-2],
'Xh_gbH2':Xh_gbH2[-2],
'Xh_gbH3':Xh_gbH3[-2],
'Xh_gbH4':Xh_gbH4[-2],
'Xh_gbH5':Xh_gbH5[-2],
'Xh_gbH6':Xh_gbH6[-2]
}

preds_sell= {
'Xl_gbl_6': Xl_gbl_6[-2],
'Xl_gbl_5': Xl_gbl_5[-2],
'Xl_gbl_4': Xl_gbl_4[-2],
'Xl_gbl_3': Xl_gbl_3[-2],
'Xl_gbl_2': Xl_gbl_2[-2],
'Xl_gbl_1': Xl_gbl_1[-2],
'Xl_gbl0': Xl_gbl0[-2],
'Xl_gbl1': Xl_gbl1[-2],
'Xl_gbl2': Xl_gbl2[-2],
'Xl_gbl3': Xl_gbl3[-2],
'Xl_gbl4': Xl_gbl4[-2],
'Xl_gbl5': Xl_gbl5[-2],
'Xl_gbl6': Xl_gbl6[-2]
}

pricediff = True
instrument = 'USD_JPY'

Actuals = ['HHLL_LogDiff {}_highAsk'.format(instrument),
           'HHLL_LogDiff {}_highBid'.format(instrument),
           'HHLL_LogDiff {}_lowAsk'.format(instrument),
           'HHLL_LogDiff {}_lowBid'.format(instrument)]

Responses = ['future diff high',
             'future diff low']

prices = [i.replace('HHLL_LogDiff ', '') for i in Actuals]
prices.append('{}_date'.format(instrument))

fxcm = gf[prices]
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].astype(str)
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].str[:13]

fxcm = fxcm.drop(0)

variableshhll = pd.read_csv('./src/assets/variables/variablesHigh.csv', index_col=0)

current_open_ask = gf['USD_JPY_openAsk'].iloc[-1].round(3)
current_open_bid = gf['USD_JPY_openBid'].iloc[-1].round(3)

# aquí se toma el precio más caro a la venta de la hora previa (investigar por qué aquí es el último dato)
previous_high_ask = gf['USD_JPY_highAsk'].iloc[-2].round(3)
take_profit_buy = [((1+i/(10000.00))*previous_high_ask).round(3) for i in range(-6, 7)]

# aquí se toma el precio más barato a la compra de la hora previa (investigar por qué aquí es el penúltimo dato)
previous_low_bid = gf['USD_JPY_lowBid'].iloc[-2].round(3)
take_profit_sell = [((1-i/(10000.00))*previous_low_bid).round(3) for i in range(-6, 7)]

op_buy = pd.DataFrame({'Open': current_open_ask,
                       'Probability':list(preds_buy.values()),
                       'Take Profit': take_profit_buy},
                  index=[f'Buy{i}' for i in range(-6, 7)])

op_sell = pd.DataFrame({'Open': current_open_bid,
                        'Probability':list(preds_sell.values()),
                        'Take Profit': take_profit_sell},
                  index=[f'Sell{i}' for i in range(-6, 7)])

op_buy.loc[op_buy['Open'] > op_buy['Take Profit'], ['Take Profit', 'Probability']] = np.nan
op_sell.loc[op_sell['Open'] < op_sell['Take Profit'], ['Take Profit', 'Probability']]  = np.nan


new = [0,2,1]
op_buy = op_buy[op_buy.columns[new]]
op_sell = op_sell[op_sell.columns[new]]

def main(argv):
    """
    Main
    """  
    TOKEN = os.environ['telegram_token']
    CHAT_ID = os.environ['telegram_chat_id']
    html_template_path ="./src/assets/email/email_template.html"

    tz_MX = pytz.timezone('America/Mexico_City') 
    datetime_MX = dt.now(tz_MX)
    
    hora_now = f'{datetime_MX.strftime("%H:%M:%S")}'

    parser = argparse.ArgumentParser(description='Apollo V 0.1 Beta')
    
    parser.add_argument('-o','--order', action='store_true',
                        help='Determine if you want to make an order')
    parser.add_argument('-t','--time', action='store_true',
                        help='Make order only if market is open')
    parser.add_argument('-d','--debug', action='store_true',
                        help='Debug mode on')

    args = parser.parse_args()
    make_order = args.order or False
    market_sensitive = args.time or False
    debug_mode = args.debug or False
    
    logging.info(f'\nMarket sensitive: {market_sensitive}')
    if market_sensitive and not market_open():
        logging.info('Market Closed')
        return

# Hacer decisón para la posición
    decision = Decide(op_buy, op_sell, 100000, direction=0, pips=1, take_profit=0 , stop_loss=0)
    decision.get_all_pips()
    units = decision.pips * decision.direction * 1000
    
    max_units = 3000 #máximo de unidades en riesgo al mismo tiempo
    positions = Positions('USD_JPY')
    positions.get_status()
    current_units = positions.long_units + positions.short_units

    if units > 0 and current_units <= max_units: # si queremos hacer una operación y aún podemos hacer operaciones
        # escogemos lo que podamos operar sin pasarnos del límite.
        units = min(abs(units), max_units - current_units) * decision.direction
        if units == 0:
            decision.decision += '\n*Unit limit exceeded. Order not placed.'
    
    inv_instrument = 'USD_JPY'
    take_profit = decision.take_profit
    op_buy_new = decision.data_buy
    op_sell_new = decision.data_sell

    logging.info(f'\n{decision.decision}')        
    # Pone orden a precio de mercado
    logging.info(f'Units: {units}, inv_instrument: {inv_instrument} , take_profit: {take_profit}\n')
        
    if make_order and units != 0:
        new_order = Order(inv_instrument, take_profit)
        new_order.make_market_order(units)

    print(f'\nPrevious High Ask:{previous_high_ask}')
    print(op_buy_new)
    print(f'\nPrevious Low Bid: {previous_low_bid}')
    print(op_sell_new)

    # send telegram
    if not debug_mode:
        _, html_path = create_html([op_buy, op_sell, previous_high_ask, previous_low_bid], html_template_path)
        _, image_name = from_html_to_jpg(html_path)
        logging.info('Se mandan predicciones a Telegram')
        bot = telegram_bot(TOKEN)
        if not make_order:
            bot.send_message(CHAT_ID, f"TEST!!!!!")
        bot.send_message(CHAT_ID, f"Predictions for the hour: {hora_now}")
        bot.send_photo(CHAT_ID, image_name)
        bot.send_message(CHAT_ID, f"Best course of action: {decision.decision}")
        
if __name__ == '__main__':
    #load settings
    with open ("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
