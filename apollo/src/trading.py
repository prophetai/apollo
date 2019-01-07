# -*- coding: utf-8 -*-

import os
import sys, getopt
import logging
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLSResults

#utolities propias
from utils import get_forex, setup_data

#librerías para manejo de tiempo
import datetime
from datetime import datetime as dt
import time
import pytz

#librerías para mandar correo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)

COMMASPACE = ', '
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
start = str(dt.now() + datetime.timedelta(days=-2))[:10]
end = str(dt.now())[:10]
freq = 'D'
trading = True

time.sleep(30) #sleep for one second.

gf = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

sd = setup_data(gf,
                instrument=instrument,
                pricediff=True,
                log=True,
                trading=True)

sd.head()

sd['intercept'] = 1



models = {}

models['HHLL_LogDiff USD_JPY_highMid0'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid0.h5')
models['HHLL_LogDiff USD_JPY_highMid1'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid1.h5')
models['HHLL_LogDiff USD_JPY_highMid2'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid2.h5')
models['HHLL_LogDiff USD_JPY_highMid3'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid3.h5')
models['HHLL_LogDiff USD_JPY_highMid4'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid4.h5')
models['HHLL_LogDiff USD_JPY_highMid5'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid5.h5')
models['HHLL_LogDiff USD_JPY_highMid6'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_highMid6.h5')

models['HHLL_LogDiff USD_JPY_lowMid0'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid0.h5')
models['HHLL_LogDiff USD_JPY_lowMid1'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid1.h5')
models['HHLL_LogDiff USD_JPY_lowMid2'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid2.h5')
models['HHLL_LogDiff USD_JPY_lowMid3'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid3.h5')
models['HHLL_LogDiff USD_JPY_lowMid4'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid4.h5')
models['HHLL_LogDiff USD_JPY_lowMid5'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid5.h5')
models['HHLL_LogDiff USD_JPY_lowMid6'] = OLSResults.load('./models/HHLL_LogDiff USD_JPY_lowMid6.h5')


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


variableshhll = pd.read_csv('./models/HHLL_variables.csv', index_col=0)

for i in Actuals:
    for k in [0,1,2,3,4,5,6]:
        i = i.replace('HHLL_', '')
        df = sd[variableshhll[i + str(k)].values.tolist()]
        x = df.values
        imod = 'HHLL_' + i + str(k)
        mod = models[imod]
        act = i.replace('HHLL_LogDiff ', '')
        fxcm['Future ' + act + str(k)] = mod.predict(x)

classpredsm = pd.DataFrame(fxcm.iloc[-2])
classpredsm.columns = ['Prices']

classpredsm = classpredsm.drop('USD_JPY_date')

new_preds = classpredsm.iloc[:2]

new_preds.columns = ['Prices']

for i in [0,1,2,3,4,5,6]:
    l = [j for j in classpredsm.index if str(i) in j]
    new_preds['p(' + str(i) + ')'] = classpredsm.loc[l]['Prices'].values
    new_preds[str(i/100) + '%'] = 0
    new_preds[str(i/100) + '%'].iloc[:2] = new_preds['Prices'] * (1+i/10000)
    new_preds[str(i/100) + '%'].iloc[-1:] = new_preds['Prices'] * (1-i/10000)

new_preds = new_preds.round(3)

new_preds = new_preds.drop('Prices', axis=1)

probas = [i for i in new_preds.columns if 'p(' in i]
prices = [i for i in new_preds.columns if '%' in i or 'P' in i]
indx = 'Buy'

op_buy = pd.DataFrame({'Take Profit': np.zeros(7)},
                  index=[indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])


op_buy['Take Profit'] = new_preds[prices].iloc[0].values
op_buy['Proba TP'] = new_preds[probas].iloc[0].values

op_buy['Stop Loss'] = new_preds[prices].iloc[1].values
op_buy['Proba SL'] = new_preds[probas].iloc[1].values

indx = 'Sell'

op_sell = pd.DataFrame({'Take Profit': np.zeros(7)},
                  index=[indx + '0',
                         indx + '1',
                         indx + '2',
                         indx + '3',
                         indx + '4',
                         indx + '5',
                         indx + '6'])

op_sell['Take Profit'] = new_preds[prices].iloc[1].values
op_sell['Proba TP'] = new_preds[probas].iloc[1].values

op_sell['Stop Loss'] = new_preds[prices].iloc[0].values
op_sell['Proba SL'] = new_preds[probas].iloc[0].values


"""# H-C/C-L"""

models = {}

models['HCL_Diff High-Close0.2'] = OLSResults.load('./models/HCL_Diff High-Close0.2.h5')
models['HCL_Diff High-Close0.4'] = OLSResults.load('./models/HCL_Diff High-Close0.4.h5')
models['HCL_Diff High-Close0.6'] = OLSResults.load('./models/HCL_Diff High-Close0.6.h5')
models['HCL_Diff High-Close1'] = OLSResults.load('./models/HCL_Diff High-Close1.h5')
models['HCL_Diff High-Close1.5'] = OLSResults.load('./models/HCL_Diff High-Close1.5.h5')
models['HCL_Diff High-Close2'] = OLSResults.load('./models/HCL_Diff High-Close2.h5')
models['HCL_Diff High-Close2.5'] = OLSResults.load('./models/HCL_Diff High-Close2.5.h5')

models['HCL_Diff Close-Low0.2'] = OLSResults.load('./models/HCL_Diff Close-Low0.2.h5')
models['HCL_Diff Close-Low0.4'] = OLSResults.load('./models/HCL_Diff Close-Low0.4.h5')
models['HCL_Diff Close-Low0.6'] = OLSResults.load('./models/HCL_Diff Close-Low0.6.h5')
models['HCL_Diff Close-Low1'] = OLSResults.load('./models/HCL_Diff Close-Low1.h5')
models['HCL_Diff Close-Low1.5'] = OLSResults.load('./models/HCL_Diff Close-Low1.5.h5')
models['HCL_Diff Close-Low2'] = OLSResults.load('./models/HCL_Diff Close-Low2.h5')
models['HCL_Diff Close-Low2.5'] = OLSResults.load('./models/HCL_Diff Close-Low2.5.h5')

pricediff = True
instrument = 'USD_JPY'

Actuals = ['Diff High-Close',
           'Diff Close-Low']

Responses = ['future High-Close',
             'future Close-Low']


prices = ['USD_JPY_highMid', 'USD_JPY_lowMid', 'USD_JPY_closeMid', 'USD_JPY_date']

gf['Diff High-Close'] = gf['USD_JPY_highMid'] - gf['USD_JPY_closeMid']
gf['Diff Close-Low'] = gf['USD_JPY_closeMid'] - gf['USD_JPY_lowMid']

fxcm = gf[prices]
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].astype(str)
fxcm['{}_date'.format(instrument)] = fxcm['{}_date'.format(instrument)].str[:13]

fxcm = fxcm.drop(0)

variableshcl = pd.read_csv('./models/HCL_variables.csv', index_col=0)

for i in Actuals:
    for k in [0.2,0.4,0.6,1,1.5,2,2.5]:
        i = i.replace('HCL_', '')
        df = sd[variableshcl[i + str(k)].values.tolist()]
        x = df.values
        imod = 'HCL_' + i + str(k)
        mod = models[imod]
        act = i.replace('HCL_Diff ', '')
        fxcm['Future ' + act + str(k)] = mod.predict(x)

classpredsm = pd.DataFrame(fxcm.iloc[-2])
classpredsm.columns = ['Prices']

classpredsm = classpredsm.drop('USD_JPY_date')

new_preds = classpredsm.iloc[:2]

new_preds.columns = ['Prices']

high_close = classpredsm.loc['USD_JPY_highMid']['Prices'] - classpredsm.loc['USD_JPY_closeMid']['Prices']
close_low = classpredsm.loc['USD_JPY_closeMid']['Prices'] - classpredsm.loc['USD_JPY_lowMid']['Prices']

buy_limits = pd.DataFrame({'Buy Limit 0.2': np.zeros(2),
                       'Proba 0.2': np.zeros(2),
                       'Buy Limit 0.4': np.zeros(2),
                       'Proba 0.4': np.zeros(2),
                       'Buy Limit 0.6': np.zeros(2),
                       'Proba 0.6': np.zeros(2),
                       'Buy Limit 1': np.zeros(2),
                       'Proba 1': np.zeros(2),
                       'Buy Limit 1.5': np.zeros(2),
                       'Proba 1.5': np.zeros(2),
                       'Buy Limit 2': np.zeros(2),
                       'Proba 2': np.zeros(2),
                       'Buy Limit 2.5': np.zeros(2),
                       'Proba 2.5': np.zeros(2)},
                     index = ['Set at', 'Take Profit'])

buy_limits['Proba 0.2'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low0.2']['Prices']
buy_limits['Proba 0.4'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low0.4']['Prices']
buy_limits['Proba 0.6'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low0.6']['Prices']
buy_limits['Proba 1'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low1']['Prices']
buy_limits['Proba 1.5'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low1.5']['Prices']
buy_limits['Proba 2'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low2']['Prices']
buy_limits['Proba 2.5'].loc['Set at'] = classpredsm.loc['Future Diff Close-Low2.5']['Prices']

buy_limits['Buy Limit 0.2'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.2
buy_limits['Buy Limit 0.4'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.4
buy_limits['Buy Limit 0.6'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.6
buy_limits['Buy Limit 1'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*1
buy_limits['Buy Limit 1.5'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*1.5
buy_limits['Buy Limit 2'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*2
buy_limits['Buy Limit 2.5'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*2.5

buy_limits['Proba 0.2'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close0.2']['Prices']
buy_limits['Proba 0.4'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close0.4']['Prices']
buy_limits['Proba 0.6'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close0.6']['Prices']
buy_limits['Proba 1'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close1']['Prices']
buy_limits['Proba 1.5'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close1.5']['Prices']
buy_limits['Proba 2'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close2']['Prices']
buy_limits['Proba 2.5'].loc['Take Profit'] = classpredsm.loc['Future Diff High-Close2.5']['Prices']

buy_limits['Buy Limit 0.2'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.2
buy_limits['Buy Limit 0.4'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.4
buy_limits['Buy Limit 0.6'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.6
buy_limits['Buy Limit 1'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*1
buy_limits['Buy Limit 1.5'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*1.5
buy_limits['Buy Limit 2'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*2
buy_limits['Buy Limit 2.5'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*2.5

sell_limits = pd.DataFrame({'Sell Limit 0.2': np.zeros(2),
                       'Proba 0.2': np.zeros(2),
                       'Sell Limit 0.4': np.zeros(2),
                       'Proba 0.4': np.zeros(2),
                       'Sell Limit 0.6': np.zeros(2),
                       'Proba 0.6': np.zeros(2),
                       'Sell Limit 1': np.zeros(2),
                       'Proba 1': np.zeros(2),
                       'Sell Limit 1.5': np.zeros(2),
                       'Proba 1.5': np.zeros(2),
                       'Sell Limit 2': np.zeros(2),
                       'Proba 2': np.zeros(2),
                       'Sell Limit 2.5': np.zeros(2),
                       'Proba 2.5': np.zeros(2)},
                     index = ['Set at', 'Take Profit'])

sell_limits['Proba 0.2'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low0.2']['Prices']
sell_limits['Proba 0.4'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low0.4']['Prices']
sell_limits['Proba 0.6'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low0.6']['Prices']
sell_limits['Proba 1'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low1']['Prices']
sell_limits['Proba 1.5'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low1.5']['Prices']
sell_limits['Proba 2'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low2']['Prices']
sell_limits['Proba 2.5'].loc['Take Profit'] = classpredsm.loc['Future Diff Close-Low2.5']['Prices']

sell_limits['Sell Limit 0.2'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.2
sell_limits['Sell Limit 0.4'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.4
sell_limits['Sell Limit 0.6'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*0.6
sell_limits['Sell Limit 1'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*1
sell_limits['Sell Limit 1.5'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*1.5
sell_limits['Sell Limit 2'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*2
sell_limits['Sell Limit 2.5'].loc['Take Profit'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] - close_low*2.5

sell_limits['Proba 0.2'].loc['Set at'] = classpredsm.loc['Future Diff High-Close0.2']['Prices']
sell_limits['Proba 0.4'].loc['Set at'] = classpredsm.loc['Future Diff High-Close0.4']['Prices']
sell_limits['Proba 0.6'].loc['Set at'] = classpredsm.loc['Future Diff High-Close0.6']['Prices']
sell_limits['Proba 1'].loc['Set at'] = classpredsm.loc['Future Diff High-Close1']['Prices']
sell_limits['Proba 1.5'].loc['Set at'] = classpredsm.loc['Future Diff High-Close1.5']['Prices']
sell_limits['Proba 2'].loc['Set at'] = classpredsm.loc['Future Diff High-Close2']['Prices']
sell_limits['Proba 2.5'].loc['Set at'] = classpredsm.loc['Future Diff High-Close2.5']['Prices']

sell_limits['Sell Limit 0.2'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.2
sell_limits['Sell Limit 0.4'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.4
sell_limits['Sell Limit 0.6'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*0.6
sell_limits['Sell Limit 1'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*1
sell_limits['Sell Limit 1.5'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*1.5
sell_limits['Sell Limit 2'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*2
sell_limits['Sell Limit 2.5'].loc['Set at'] = classpredsm.loc['USD_JPY_closeMid']['Prices'] + high_close*2.5


"""## Gameplan"""

op_sell['Sell Limits'] = np.nan
op_sell['Proba Sell Limits'] = np.nan
op_sell['Sell Limits'].iloc[0] = sell_limits['Sell Limit 0.2'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[0] = sell_limits['Proba 0.2'].loc['Set at']
op_sell['Sell Limits'].iloc[1] = sell_limits['Sell Limit 0.4'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[1] = sell_limits['Proba 0.4'].loc['Set at']
op_sell['Sell Limits'].iloc[2] = sell_limits['Sell Limit 0.6'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[2] = sell_limits['Proba 0.6'].loc['Set at']
op_sell['Sell Limits'].iloc[3] = sell_limits['Sell Limit 1'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[3] = sell_limits['Proba 1'].loc['Set at']
op_sell['Sell Limits'].iloc[4] = sell_limits['Sell Limit 1.5'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[4] = sell_limits['Proba 1.5'].loc['Set at']
op_sell['Sell Limits'].iloc[5] = sell_limits['Sell Limit 2'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[5] = sell_limits['Proba 2'].loc['Set at']
op_sell['Sell Limits'].iloc[6] = sell_limits['Sell Limit 2.5'].loc['Set at']
op_sell['Proba Sell Limits'].iloc[6] = sell_limits['Proba 2.5'].loc['Set at']

op_buy['Buy Limits'] = np.nan
op_buy['Proba Buy Limits'] = np.nan
op_buy['Buy Limits'].iloc[0] = buy_limits['Buy Limit 0.2'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[0] = buy_limits['Proba 0.2'].loc['Set at']
op_buy['Buy Limits'].iloc[1] = buy_limits['Buy Limit 0.4'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[1] = buy_limits['Proba 0.4'].loc['Set at']
op_buy['Buy Limits'].iloc[2] = buy_limits['Buy Limit 0.6'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[2] = buy_limits['Proba 0.6'].loc['Set at']
op_buy['Buy Limits'].iloc[3] = buy_limits['Buy Limit 1'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[3] = buy_limits['Proba 1'].loc['Set at']
op_buy['Buy Limits'].iloc[4] = buy_limits['Buy Limit 1.5'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[4] = buy_limits['Proba 1.5'].loc['Set at']
op_buy['Buy Limits'].iloc[5] = buy_limits['Buy Limit 2'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[5] = buy_limits['Proba 2'].loc['Set at']
op_buy['Buy Limits'].iloc[6] = buy_limits['Buy Limit 2.5'].loc['Set at']
op_buy['Proba Buy Limits'].iloc[6] = buy_limits['Proba 2.5'].loc['Set at']


op_buy = op_buy.drop(['Stop Loss', 'Proba SL'], axis=1).round(3)
op_sell = op_sell.drop(['Stop Loss', 'Proba SL'], axis=1).round(3)


def send_email(subject,fromaddr, toaddr, password,body_text):
    """
    Manda email de tu correo a tu correo
    Args:
        subject (str): Asunto del correo
        body_test (str): Cuerpo del correo
    """

    toaddr = toaddr.split(' ')
    html_template = open("./email/email_template.html", 'r')
    html_template = html_template.read()

    # datetime object with timezone awareness:
    dt.now(tz=pytz.utc)

    # seconds from epoch:
    dt.now(tz=pytz.utc).timestamp()

    # ms from epoch:
    hora_now = int(dt.now(tz=pytz.utc).timestamp() * 1000)
    hora_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    msg = MIMEMultipart()
    msg.preamble = f'Predicciones de la hora {hora_now}'
    msg['From'] = fromaddr
    msg['To'] = COMMASPACE.join(toaddr)
    msg['Subject'] = subject + ' '+str(hora_now)

    soup = BeautifulSoup(html_template, features="lxml")
    find_buy = soup.find("table", {"id": "buy_table"})
    br = soup.new_tag('br')

    for i, table in enumerate(soup.select('table.dataframe')):
        print(f'i: {i}')
        print(f'body_text[{i}]: {body_text[i]}')
        table.replace_with(BeautifulSoup(body_text[i].to_html(), "html.parser"))


    msg.attach(MIMEText(soup, 'html'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def main(argv):
    """
    Main
    """
    stackdriver_logging = False
    try:
      opts, args = getopt.getopt(argv,"l:",["logging"])
    except getopt.GetoptError:
        print('trading.py -l')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('hola!!')
            print('automl.py -l')
            sys.exit(2)
        elif opt in ("-l", "--logging"):
            stackdriver_logging = True

    if stackdriver_logging:
        import google.cloud.logging
        # Inicia el cliente de logs de Google
        logging_client = google.cloud.logging.Client()
        # Configura que todos los logs se vayan a stackdriver
        logging_client.setup_logging()

    #send emails
    send_email('Predicciones de USDJPY',
                os.environ['email_from'],
                os.environ['email_members'],
                os.environ['email_pass'],
                [op_buy, op_sell])

if __name__ == '__main__':
    #load settings
    with open ("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
