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

    def loadData(self):
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
        time.sleep(3)
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
        Xh, Xl, original_dataset = self.loadData()

        models = {}

        high_six_minus = model_files[:6]
        high_six_minus.reverse()

        hig_six_plus = model_files[6:13]
        hig_six_plus.reverse()

        low_six_minus = model_files[13:19]
        low_six_minus.reverse()

        low_six_plus = model_files[19:]
        low_six_plus.reverse()

        logging.info(f'Xh:{len(Xh[-1])}')
        logging.info('************* Haciendo predicciones **************')

        for i, file_model in enumerate(high_six_minus):
            models[f'Xh_gbH_{i+1}'] = file_model.predict_proba(Xh)[:, 1]

        for i, file_model in enumerate(hig_six_plus):
            models[f'Xh_gbH{i}'] = file_model.predict_proba(Xh)[:, 1]

        for i, file_model in enumerate(low_six_minus):
            models[f'Xl_gbl_{i+1}'] = file_model.predict_proba(Xl)[:, 1]

        for i, file_model in enumerate(low_six_plus):
            models[f'Xl_gbl{i}'] = file_model.predict_proba(Xl)[:, 1]
        #
        #models.update({f'Xl_gbl_{i}': file_model.predict_proba(Xl)[:,1] for i, file_model in enumerate(model_files[len(model_files)/2:])})

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
        previous_high_ask = original_dataset['USD_JPY_highAsk'].iloc[-2].round(
            3)
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

        return op_buy, op_sell, original_dataset
