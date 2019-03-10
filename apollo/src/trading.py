
# coding: utf-8

import argparse
import pytz
import os
import sys, getopt
import logging
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLSResults

from send_predictions.email_send import send_email, create_html, from_html_to_jpg
from send_predictions.telegram_send import telegram_bot


#utolities propias
from utils import get_forex, setup_data

#librerías para manejo de tiempo
import datetime
from datetime import datetime as dt
import time

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)

candleformat = 'midpoint'
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
start = str(datetime.datetime.now() + datetime.timedelta(days=-2))[:10]
end = str(dt.now())[:10]
print(f'Data from start:{start}, end:{end}')
freq = 'D'
trading = True

#obtenemos los datos del instrumento principal y de los adicionales
time.sleep(10) #damos oportunidad a OANDA de que tenga los datos que necesitamos
gf = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

sd = setup_data(gf,
                instrument=instrument,
                pricediff=True,
                log=True,
                trading=True)

sd.head()

sd['intercept'] = 1




models = {}

models['HHLL_LogDiff USD_JPY_highMid-1'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-1.h5')
models['HHLL_LogDiff USD_JPY_highMid-2'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-2.h5')
models['HHLL_LogDiff USD_JPY_highMid-3'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-3.h5')
models['HHLL_LogDiff USD_JPY_highMid-4'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-4.h5')
models['HHLL_LogDiff USD_JPY_highMid-5'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-5.h5')
models['HHLL_LogDiff USD_JPY_highMid-6'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid-6.h5')

models['HHLL_LogDiff USD_JPY_lowMid-1'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-1.h5')
models['HHLL_LogDiff USD_JPY_lowMid-2'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-2.h5')
models['HHLL_LogDiff USD_JPY_lowMid-3'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-3.h5')
models['HHLL_LogDiff USD_JPY_lowMid-4'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-4.h5')
models['HHLL_LogDiff USD_JPY_lowMid-5'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-5.h5')
models['HHLL_LogDiff USD_JPY_lowMid-6'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid-6.h5')

models['HHLL_LogDiff USD_JPY_highMid0'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid0.h5')
models['HHLL_LogDiff USD_JPY_highMid1'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid1.h5')
models['HHLL_LogDiff USD_JPY_highMid2'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid2.h5')
models['HHLL_LogDiff USD_JPY_highMid3'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid3.h5')
models['HHLL_LogDiff USD_JPY_highMid4'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid4.h5')
models['HHLL_LogDiff USD_JPY_highMid5'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid5.h5')
models['HHLL_LogDiff USD_JPY_highMid6'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_highMid6.h5')

models['HHLL_LogDiff USD_JPY_lowMid0'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid0.h5')
models['HHLL_LogDiff USD_JPY_lowMid1'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid1.h5')
models['HHLL_LogDiff USD_JPY_lowMid2'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid2.h5')
models['HHLL_LogDiff USD_JPY_lowMid3'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid3.h5')
models['HHLL_LogDiff USD_JPY_lowMid4'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid4.h5')
models['HHLL_LogDiff USD_JPY_lowMid5'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid5.h5')
models['HHLL_LogDiff USD_JPY_lowMid6'] = OLSResults.load('./src/models/HHLL_LogDiff USD_JPY_lowMid6.h5')


pricediff = True
instrument = 'USD_JPY'

Actuals = ['HHLL_LogDiff {}_highMid'.format(instrument),
           'HHLL_LogDiff {}_lowMid'.format(instrument)]

Responses = ['future diff high',
             'future diff low']

prices = [i.replace('HHLL_LogDiff ', '') for i in Actuals]
prices.append('{}_date'.format(instrument))

fxcm = gf[prices]
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].astype(str)
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].str[:13]

fxcm = fxcm.drop(0)

variableshhll = pd.read_csv('./src/models/HHLL_variables.csv', index_col=0)

for i in Actuals:
    for k in [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]:
        i = i.replace('HHLL_', '')
        var = variableshhll[i + str(k)].dropna().values.tolist()
        x = sd[var].values
        imod = 'HHLL_' + i + str(k)
        mod = models[imod]
        act = i.replace('HHLL_LogDiff ', '')
        fxcm['Future ' + act + str(k)] = mod.predict(x)

classpredsm = pd.DataFrame(fxcm.iloc[-2])
classpredsm.columns = ['Prices']

classpredsm = classpredsm.drop('USD_JPY_date')


# In[3]:


high = [i for i in classpredsm.index if 'high' in i and 'Future' in i]
high = classpredsm['Prices'].loc[high]
low = [i for i in classpredsm.index if 'low' in i and 'Future' in i]
low = classpredsm['Prices'].loc[low]


new_preds = classpredsm.iloc[:2]

new_preds.columns = ['Prices']
k = 0
for i in [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]:
    l = [j for j in classpredsm.index if str(i) in j]
    new_preds['p(' + str(i) + ')'] = 0
    new_preds['p(' + str(i) + ')'].iloc[0] = high.values[k]
    new_preds['p(' + str(i) + ')'].iloc[1] = low.values[k]
    new_preds[str(i/100) + '%'] = 0
    new_preds[str(i/100) + '%'].iloc[0] = new_preds['Prices'].iloc[0] * (1+i/10000)
    new_preds[str(i/100) + '%'].iloc[1] = new_preds['Prices'].iloc[1] * (1-i/10000)
    k += 1

new_preds = new_preds.round(3)

new_preds = new_preds.drop('Prices', axis=1)


probas = [i for i in new_preds.columns if 'p(' in i]
prices = [i for i in new_preds.columns if '%' in i or 'P' in i]


indx = 'Buy'
current_open = gf['USD_JPY_openMid'].iloc[-1].round(3)

op_buy = pd.DataFrame({'Take Profit': np.zeros(13)},
                  index=[indx + '-6',
                         indx + '-5',
                         indx + '-4',
                         indx + '-3',
                         indx + '-2',
                         indx + '-1',
                         indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])

op_buy['Take Profit'] = new_preds[prices].iloc[0].values
op_buy['Probability'] = new_preds[probas].iloc[0].values

indx = 'Sell'

op_sell = pd.DataFrame({'Take Profit': np.zeros(13)},
                  index=[indx + '-6',
                         indx + '-5',
                         indx + '-4',
                         indx + '-3',
                         indx + '-2',
                         indx + '-1',
                         indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])

op_sell['Take Profit'] = new_preds[prices].iloc[1].values
op_sell['Probability'] = new_preds[probas].iloc[1].values

op_buy['Open'] = current_open
op_sell['Open'] = current_open

op_buy.loc[op_buy['Open'] > op_buy['Take Profit'], ['Take Profit', 'Probability']] = np.nan
op_sell.loc[op_sell['Open'] < op_sell['Take Profit'], ['Take Profit', 'Probability']]  = np.nan

opb = op_buy[:6]
opb2 = op_buy[6:]
opb.loc['Buy_'] = np.nan

opb['Sell Limit'] = opb2['Take Profit'].values
opb['Sell Limit Probability'] = opb2['Probability'].values

opb.index = opb2.index

ops = op_sell[:6]
ops2 = op_sell[6:]
ops.loc['Buy_'] = np.nan

ops['Buy Limit'] = ops2['Take Profit'].values
ops['Buy Limit Probability'] = ops2['Probability'].values

ops.index = ops2.index

new_order = [2,0,1,3,4]
opb = opb[opb.columns[new_order]]
ops = ops[ops.columns[new_order]]



op_buy['Profit 0.01'] = 100*(op_buy['Take Profit'] - op_buy['Open'])
op_sell['Profit 0.01'] = 100*(-op_sell['Take Profit'] + op_sell['Open'])


new = [2,0,1,3]
op_buy = op_buy[op_buy.columns[new]]
op_sell = op_sell[op_sell.columns[new]]


buckets = pd.read_csv('./src/models/HHLL_buckets.csv', index_col=0)
highs = [c for c in buckets.columns if 'high' in c]
lows = [c for c in buckets.columns if 'low' in c]

op_buy['bucket'] = np.nan

def mapper(x):
    res = np.nan
    if x < 0.1:
        res = '0-10'
    if x >= 0.1 and x < 0.2:
        res = '10-20'
    if x >= 0.2 and x < 0.3:
        res = '20-30'
    if x >= 0.3 and x < 0.4:
        res = '30-40'
    if x >= 0.4 and x < 0.5:
        res = '40-50'
    if x >= 0.5 and x < 0.6:
        res = '50-60'
    if x >= 0.6 and x < 0.7:
        res = '60-70'
    if x >= 0.7 and x < 0.8:
        res = '70-80'
    if x >= 0.8 and x < 0.9:
        res = '80-90'
    if x >= 0.9:
        res = '90-100'
    return res

for i in range(len(op_buy)):
    bucket_col = buckets[buckets[highs].columns[i]]
    ind = mapper(op_buy.iloc[i]['Probability'])
    try:
        op_buy['bucket'].iloc[i] = bucket_col.loc[ind]
    except:
        pass

op_sell['bucket'] = np.nan
for i in range(len(op_sell)):
    bucket_col = buckets[buckets[lows].columns[i]]
    ind = mapper(op_sell.iloc[i]['Probability'])
    try:
        op_sell['bucket'].iloc[i] = bucket_col.loc[ind]
    except:
        pass
op_buy.drop(columns=['Profit 0.01'], inplace=True)
op_sell.drop(columns=['Profit 0.01'], inplace=True)
print(op_buy)
print(op_sell)


def main(argv):
    """
    Main
    """
    TOKEN = os.environ['telegram_token']
    CHAT_ID = os.environ['telegram_chat_id']
    html_template_path ="./src/assets/email/email_template.html"

    hora_now = f'{datetime.datetime.now() - datetime.timedelta(hours=6):%Y-%m-%d %H:%M:%S}'

    parser = argparse.ArgumentParser(description='Apollo V 0.1 Beta')
    parser.add_argument('-o','--order', action='store_true',
                        help='Determine if you want to make an order')

    args = parser.parse_args()
    make_order = args.order or False

    if make_order:
        # Hacer decisón para la posición

        # Pone orden a precio de mercado
        new_order = Order()
        new_order.make_market_order(units, inv_instrument)
        # Poner stop loss de orden
        new_order.set_stop_loss(stop_loss)
        # Poner Take profit
        new_order.set_take_profit(take_profit)

    html_file, html_path = create_html([op_buy, op_sell], html_template_path)
    image_file, image_name = from_html_to_jpg(html_path)

    # send emails
    send_email('Predicciones de USDJPY',
                os.environ['email_from'],
                os.environ['email_members'],
                os.environ['email_pass'],
                html_file)

    # send telegram
    bot = telegram_bot(TOKEN)
    bot.send_message(CHAT_ID, f"Predicciones de la hora {hora_now}")
    bot.send_photo(CHAT_ID, f'{image_name}')

if __name__ == '__main__':
    #load settings
    with open ("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
