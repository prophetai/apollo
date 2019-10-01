import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime
import pickle
import sys
from sklearn import preprocessing
from datetime import datetime as dt
from keras.models import load_model
sys.path.append('./src/assets/')
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators



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
start = str(datetime.datetime.now() + datetime.timedelta(days=-2))[:10]
end = str(dt.now())[:10]
freq = 'D'
trading = True


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

variablesh = pd.read_csv('./src/assets/variables/variablesHigh.csv')
variablesh = list(variablesh['0'].values)

variablesl = pd.read_csv('./src/assets/variables/variablesLow.csv')
variablesl = list(variablesl['0'].values)


Xh = processeddf[variablesh]
scaler = pickle.load(open('./src/assets/models/scalerH', 'rb'))
Xh = scaler.transform(Xh)


Xl = processeddf[variablesl]
scaler = pickle.load(open('./src/assets/models/scalerL', 'rb'))
Xl = scaler.transform(Xl)

lrH_6 = pickle.load(open('./src/assets/models/lrHigh-6.h5', 'rb'))
lrH_5 = pickle.load(open('./src/assets/models/lrHigh-5.h5', 'rb'))
lrH_4 = pickle.load(open('./src/assets/models/lrHigh-4.h5', 'rb'))
lrH_3 = pickle.load(open('./src/assets/models/lrHigh-3.h5', 'rb'))
lrH_2 = pickle.load(open('./src/assets/models/lrHigh-2.h5', 'rb'))
lrH_1 = pickle.load(open('./src/assets/models/lrHigh-1.h5', 'rb'))
lrH0 = pickle.load(open('./src/assets/models/lrHigh0.h5', 'rb'))
lrH1 = pickle.load(open('./src/assets/models/lrHigh1.h5', 'rb'))
lrH2 = pickle.load(open('./src/assets/models/lrHigh2.h5', 'rb'))
lrH3 = pickle.load(open('./src/assets/models/lrHigh3.h5', 'rb'))
lrH4 = pickle.load(open('./src/assets/models/lrHigh4.h5', 'rb'))
lrH5 = pickle.load(open('./src/assets/models/lrHigh5.h5', 'rb'))
lrH6 = pickle.load(open('./src/assets/models/lrHigh6.h5', 'rb'))

rfH_6 = pickle.load(open('./src/assets/models/rfHigh-6.h5', 'rb'))
rfH_5 = pickle.load(open('./src/assets/models/rfHigh-5.h5', 'rb'))
rfH_4 = pickle.load(open('./src/assets/models/rfHigh-4.h5', 'rb'))
rfH_3 = pickle.load(open('./src/assets/models/rfHigh-3.h5', 'rb'))
rfH_2 = pickle.load(open('./src/assets/models/rfHigh-2.h5', 'rb'))
rfH_1 = pickle.load(open('./src/assets/models/rfHigh-1.h5', 'rb'))
rfH0 = pickle.load(open('./src/assets/models/rfHigh0.h5', 'rb'))
rfH1 = pickle.load(open('./src/assets/models/rfHigh1.h5', 'rb'))
rfH2 = pickle.load(open('./src/assets/models/rfHigh2.h5', 'rb'))
rfH3 = pickle.load(open('./src/assets/models/rfHigh3.h5', 'rb'))
rfH4 = pickle.load(open('./src/assets/models/rfHigh4.h5', 'rb'))
rfH5 = pickle.load(open('./src/assets/models/rfHigh5.h5', 'rb'))
rfH6 = pickle.load(open('./src/assets/models/rfHigh6.h5', 'rb'))

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

nnH_6 = load_model('./src/assets/models/nnHigh-6.h5')
nnH_5 = load_model('./src/assets/models/nnHigh-5.h5')
nnH_4 = load_model('./src/assets/models/nnHigh-4.h5')
nnH_3 = load_model('./src/assets/models/nnHigh-3.h5')
nnH_2 = load_model('./src/assets/models/nnHigh-2.h5')
nnH_1 = load_model('./src/assets/models/nnHigh-1.h5')
nnH0 = load_model('./src/assets/models/nnHigh0.h5')
nnH1 = load_model('./src/assets/models/nnHigh1.h5')
nnH2 = load_model('./src/assets/models/nnHigh2.h5')
nnH3 = load_model('./src/assets/models/nnHigh3.h5')
nnH4 = load_model('./src/assets/models/nnHigh4.h5')
nnH5 = load_model('./src/assets/models/nnHigh5.h5')
nnH6 = load_model('./src/assets/models/nnHigh6.h5')

lrL_6 = pickle.load(open('./src/assets/models/lrLow-6.h5', 'rb'))
lrL_5 = pickle.load(open('./src/assets/models/lrLow-5.h5', 'rb'))
lrL_4 = pickle.load(open('./src/assets/models/lrLow-4.h5', 'rb'))
lrL_3 = pickle.load(open('./src/assets/models/lrLow-3.h5', 'rb'))
lrL_2 = pickle.load(open('./src/assets/models/lrLow-2.h5', 'rb'))
lrL_1 = pickle.load(open('./src/assets/models/lrLow-1.h5', 'rb'))
lrL0 = pickle.load(open('./src/assets/models/lrLow0.h5', 'rb'))
lrL1 = pickle.load(open('./src/assets/models/lrLow1.h5', 'rb'))
lrL2 = pickle.load(open('./src/assets/models/lrLow2.h5', 'rb'))
lrL3 = pickle.load(open('./src/assets/models/lrLow3.h5', 'rb'))
lrL4 = pickle.load(open('./src/assets/models/lrLow4.h5', 'rb'))
lrL5 = pickle.load(open('./src/assets/models/lrLow5.h5', 'rb'))
lrL6 = pickle.load(open('./src/assets/models/lrLow6.h5', 'rb'))

rfL_6 = pickle.load(open('./src/assets/models/rfLow-6.h5', 'rb'))
rfL_5 = pickle.load(open('./src/assets/models/rfLow-5.h5', 'rb'))
rfL_4 = pickle.load(open('./src/assets/models/rfLow-4.h5', 'rb'))
rfL_3 = pickle.load(open('./src/assets/models/rfLow-3.h5', 'rb'))
rfL_2 = pickle.load(open('./src/assets/models/rfLow-2.h5', 'rb'))
rfL_1 = pickle.load(open('./src/assets/models/rfLow-1.h5', 'rb'))
rfL0 = pickle.load(open('./src/assets/models/rfLow0.h5', 'rb'))
rfL1 = pickle.load(open('./src/assets/models/rfLow1.h5', 'rb'))
rfL2 = pickle.load(open('./src/assets/models/rfLow2.h5', 'rb'))
rfL3 = pickle.load(open('./src/assets/models/rfLow3.h5', 'rb'))
rfL4 = pickle.load(open('./src/assets/models/rfLow4.h5', 'rb'))
rfL5 = pickle.load(open('./src/assets/models/rfLow5.h5', 'rb'))
rfL6 = pickle.load(open('./src/assets/models/rfLow6.h5', 'rb'))

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

nnL_6 = load_model('./src/assets/models/nnLow-6.h5')
nnL_5 = load_model('./src/assets/models/nnLow-5.h5')
nnL_4 = load_model('./src/assets/models/nnLow-4.h5')
nnL_3 = load_model('./src/assets/models/nnLow-3.h5')
nnL_2 = load_model('./src/assets/models/nnLow-2.h5')
nnL_1 = load_model('./src/assets/models/nnLow-1.h5')
nnL0 = load_model('./src/assets/models/nnLow0.h5')
nnL1 = load_model('./src/assets/models/nnLow1.h5')
nnL2 = load_model('./src/assets/models/nnLow2.h5')
nnL3 = load_model('./src/assets/models/nnLow3.h5')
nnL4 = load_model('./src/assets/models/nnLow4.h5')
nnL5 = load_model('./src/assets/models/nnLow5.h5')
nnL6 = load_model('./src/assets/models/nnLow6.h5')


Xh_lrH_6 = lrH_6.predict(Xh)
Xh_lrH_5 = lrH_5.predict(Xh)
Xh_lrH_4 = lrH_4.predict(Xh)
Xh_lrH_3 = lrH_3.predict(Xh)
Xh_lrH_2 = lrH_2.predict(Xh)
Xh_lrH_1 = lrH_1.predict(Xh)
Xh_lrH0 = lrH0.predict(Xh)
Xh_lrH1 = lrH1.predict(Xh)
Xh_lrH2 = lrH2.predict(Xh)
Xh_lrH3 = lrH3.predict(Xh)
Xh_lrH4 = lrH4.predict(Xh)
Xh_lrH5 = lrH5.predict(Xh)
Xh_lrH6 = lrH6.predict(Xh)

Xh_rfH_6 = rfH_6.predict_proba(Xh)[:,1]
Xh_rfH_5 = rfH_5.predict_proba(Xh)[:,1]
Xh_rfH_4 = rfH_4.predict_proba(Xh)[:,1]
Xh_rfH_3 = rfH_3.predict_proba(Xh)[:,1]
Xh_rfH_2 = rfH_2.predict_proba(Xh)[:,1]
Xh_rfH_1 = rfH_1.predict_proba(Xh)[:,1]
Xh_rfH0 = rfH0.predict_proba(Xh)[:,1]
Xh_rfH1 = rfH1.predict_proba(Xh)[:,1]
Xh_rfH2 = rfH2.predict_proba(Xh)[:,1]
Xh_rfH3 = rfH3.predict_proba(Xh)[:,1]
Xh_rfH4 = rfH4.predict_proba(Xh)[:,1]
Xh_rfH5 = rfH5.predict_proba(Xh)[:,1]
Xh_rfH6 = rfH6.predict_proba(Xh)[:,1]

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

Xh_nnH_6 = nnH_6.predict(Xh)
Xh_nnH_5 = nnH_5.predict(Xh)
Xh_nnH_4 = nnH_4.predict(Xh)
Xh_nnH_3 = nnH_3.predict(Xh)
Xh_nnH_2 = nnH_2.predict(Xh)
Xh_nnH_1 = nnH_1.predict(Xh)
Xh_nnH0 = nnH0.predict(Xh)
Xh_nnH1 = nnH1.predict(Xh)
Xh_nnH2 = nnH2.predict(Xh)
Xh_nnH3 = nnH3.predict(Xh)
Xh_nnH4 = nnH4.predict(Xh)
Xh_nnH5 = nnH5.predict(Xh)
Xh_nnH6 = nnH6.predict(Xh)

Xh_meanH_6 = (Xh_lrH_6 + Xh_rfH_6 + Xh_gbH_6 + Xh_nnH_6.T[0])/4
Xh_meanH_5 = (Xh_lrH_5 + Xh_rfH_5 + Xh_gbH_5 + Xh_nnH_5.T[0])/4
Xh_meanH_4 = (Xh_lrH_4 + Xh_rfH_4 + Xh_gbH_4 + Xh_nnH_4.T[0])/4
Xh_meanH_3 = (Xh_lrH_3 + Xh_rfH_3 + Xh_gbH_3 + Xh_nnH_3.T[0])/4
Xh_meanH_2 = (Xh_lrH_2 + Xh_rfH_2 + Xh_gbH_2 + Xh_nnH_2.T[0])/4
Xh_meanH_1 = (Xh_lrH_1 + Xh_rfH_1 + Xh_gbH_1 + Xh_nnH_1.T[0])/4
Xh_meanH0 = (Xh_lrH0 + Xh_rfH0 + Xh_gbH0 + Xh_nnH0.T[0])/4
Xh_meanH1 = (Xh_lrH1 + Xh_rfH1 + Xh_gbH1 + Xh_nnH1.T[0])/4
Xh_meanH2 = (Xh_lrH2 + Xh_rfH2 + Xh_gbH2 + Xh_nnH2.T[0])/4
Xh_meanH3 = (Xh_lrH3 + Xh_rfH3 + Xh_gbH3 + Xh_nnH3.T[0])/4
Xh_meanH4 = (Xh_lrH4 + Xh_rfH4 + Xh_gbH4 + Xh_nnH4.T[0])/4
Xh_meanH5 = (Xh_lrH5 + Xh_rfH5 + Xh_gbH5 + Xh_nnH5.T[0])/4
Xh_meanH6 = (Xh_lrH6 + Xh_rfH6 + Xh_gbH6 + Xh_nnH6.T[0])/4

gf = get_forex(instrument, instruments, granularity, start, end, candleformat, freq, trading)

sd = setup_data(gf,
                instrument=instrument,
                pricediff=True,
                log=True,
                trading=True)