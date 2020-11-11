#!/usr/bin/env python
# coding: utf-8

from data import Data
from processData.processing import setup_data, get_indicators
from loadAssets import Assets
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
from utils import truncate
from check_market import market_open
# librerías para manejo de tiempo
from datetime import timedelta
from datetime import datetime as dt
from datetime import date
import time
import math

import warnings
warnings.filterwarnings("ignore")
# sys.path.append('./src/assets/')
# sys.path.append('./src')

pd.options.display.max_columns = None

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='/tmp/log_test.txt'
)


class Trading():
    # constantes para extraer datos
    logging.info('\n\n************* Inicio **************')
    
    def __init__(self, model_version, instrument):
        self.model_version = model_version
        self.instrument = instrument

    def loadAssets(self):
        # Carga de assets
        logging.info('************* Carga de modelos **************')
        assets = Assets(self.model_version, self.instrument)
        model_files = assets.load_models()

        return model_files

    def loadData(self, delay=0):
        time.sleep(delay)
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

        # Descarga de datos
        logging.info('************* Descargando datos **************')
        data = Data(instrument=self.instrument,
                    ins_variables=instruments,
                    granularity='H1',
                    start=str(dt.now() + timedelta(days=-2))[:10],
                    end=str(dt.now())[:10],
                    freq='D',
                    candleformat='bidask',
                    trading=True)

        Xh, Xl, original_dataset = data.get_data(self.model_version)

        return Xh, Xl, original_dataset
    
    def predict(self):
        # HAY QUE VER CÓMO ARREGLAR PARA PONER DE -6 A 6 Y ERROR QUE SALE DEL LINTER
        model_files = self.loadAssets()
        Xh, Xl, original_dataset = self.loadData(delay=3)

        preds = {}

        logging.info(f'# of input variables:{len(Xh[-1])}')
        logging.info('************* Haciendo predicciones **************')

        preds['gbHigh-6'] = model_files['gbHigh-6'].predict_proba(Xh)[:, 1]
        preds['gbHigh-5'] = model_files['gbHigh-5'].predict_proba(Xh)[:, 1]
        preds['gbHigh-4'] = model_files['gbHigh-4'].predict_proba(Xh)[:, 1]
        preds['gbHigh-3'] = model_files['gbHigh-3'].predict_proba(Xh)[:, 1]        
        preds['gbHigh-2'] = model_files['gbHigh-2'].predict_proba(Xh)[:, 1]
        preds['gbHigh-1'] = model_files['gbHigh-1'].predict_proba(Xh)[:, 1]
        preds['gbHigh0'] = model_files['gbHigh0'].predict_proba(Xh)[:, 1]
        preds['gbHigh1'] = model_files['gbHigh1'].predict_proba(Xh)[:, 1]
        preds['gbHigh2'] = model_files['gbHigh2'].predict_proba(Xh)[:, 1]
        preds['gbHigh3'] = model_files['gbHigh3'].predict_proba(Xh)[:, 1]
        preds['gbHigh4'] = model_files['gbHigh4'].predict_proba(Xh)[:, 1]
        preds['gbHigh5'] = model_files['gbHigh5'].predict_proba(Xh)[:, 1]
        preds['gbHigh6'] = model_files['gbHigh6'].predict_proba(Xh)[:, 1]
        
        preds['gbLow-6'] = model_files['gbLow-6'].predict_proba(Xl)[:, 1]
        preds['gbLow-5'] = model_files['gbLow-5'].predict_proba(Xl)[:, 1]
        preds['gbLow-4'] = model_files['gbLow-4'].predict_proba(Xl)[:, 1]
        preds['gbLow-3'] = model_files['gbLow-3'].predict_proba(Xl)[:, 1]
        preds['gbLow-2'] = model_files['gbLow-2'].predict_proba(Xl)[:, 1]
        preds['gbLow-1'] = model_files['gbLow-1'].predict_proba(Xl)[:, 1]
        preds['gbLow0'] = model_files['gbLow0'].predict_proba(Xl)[:, 1]
        preds['gbLow1'] = model_files['gbLow1'].predict_proba(Xl)[:, 1]
        preds['gbLow2'] = model_files['gbLow2'].predict_proba(Xl)[:, 1]
        preds['gbLow3'] = model_files['gbLow3'].predict_proba(Xl)[:, 1]
        preds['gbLow4'] = model_files['gbLow4'].predict_proba(Xl)[:, 1]
        preds['gbLow5'] = model_files['gbLow5'].predict_proba(Xl)[:, 1]
        preds['gbLow6'] = model_files['gbLow6'].predict_proba(Xl)[:, 1]


                

        preds_buy = {
            'gbHigh-6': round(preds['gbHigh-6'][-2],2),
            'gbHigh-5': round(preds['gbHigh-5'][-2],2),
            'gbHigh-4': round(preds['gbHigh-4'][-2],2),
            'gbHigh-3': round(preds['gbHigh-3'][-2],2),
            'gbHigh-2': round(preds['gbHigh-2'][-2],2),
            'gbHigh-1': round(preds['gbHigh-1'][-2],2),
            'gbHigh0': round(preds['gbHigh0'][-2],2),
            'gbHigh1': round(preds['gbHigh1'][-2],2),
            'gbHigh2': round(preds['gbHigh2'][-2],2),
            'gbHigh3': round(preds['gbHigh3'][-2],2),
            'gbHigh4': round(preds['gbHigh4'][-2],2),
            'gbHigh5': round(preds['gbHigh5'][-2],2),
            'gbHigh6': round(preds['gbHigh6'][-2],2)
        }

        preds_sell = {
            'gbLow-6': round(preds['gbLow-6'][-2],2),
            'gbLow-5': round(preds['gbLow-5'][-2],2),
            'gbLow-4': round(preds['gbLow-4'][-2],2),
            'gbLow-3': round(preds['gbLow-3'][-2],2),
            'gbLow-2': round(preds['gbLow-2'][-2],2),
            'gbLow-1': round(preds['gbLow-1'][-2],2),
            'gbLow0': round(preds['gbLow0'][-2],2),
            'gbLow1': round(preds['gbLow1'][-2],2),
            'gbLow2': round(preds['gbLow2'][-2],2),
            'gbLow3': round(preds['gbLow3'][-2],2),
            'gbLow4': round(preds['gbLow4'][-2],2),
            'gbLow5': round(preds['gbLow5'][-2],2),
            'gbLow6': round(preds['gbLow6'][-2],2)
        }

        Actuals = ['HHLL_LogDiff {}_highAsk'.format(self.instrument),
                   'HHLL_LogDiff {}_highBid'.format(self.instrument),
                   'HHLL_LogDiff {}_lowAsk'.format(self.instrument),
                   'HHLL_LogDiff {}_lowBid'.format(self.instrument)]

        prices = [i.replace('HHLL_LogDiff ', '') for i in Actuals]
        prices.append('{}_date'.format(self.instrument))

        fxcm = original_dataset[prices]
        fxcm['{}_date'.format(self.instrument)] = fxcm['{}_date'.format(
            self.instrument)].astype(str)
        fxcm['{}_date'.format(self.instrument)] = fxcm['{}_date'.format(
            self.instrument)].str[:13]

        fxcm = fxcm.drop(0)

        current_open_ask = original_dataset['USD_JPY_openAsk'].iloc[-1].round(
            3)
        current_open_bid = original_dataset['USD_JPY_openBid'].iloc[-1].round(
            3)
        

        # aquí se toma el precio más caro a la venta de la hora previa (investigar por qué aquí es el último dato)
        previous_high_ask = original_dataset['USD_JPY_highAsk'].iloc[-2].round(3)
        take_profit_buy = [((1+i/(10000.00))*previous_high_ask).round(3)	
                           for i in range(-6, 7)]

        # aquí se toma el precio más barato a la compra de la hora previa (investigar por qué aquí es el penúltimo dato)
        previous_low_bid = original_dataset['USD_JPY_lowBid'].iloc[-2].round(3)
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

        return op_buy, op_sell, [Xh, Xl, original_dataset]
