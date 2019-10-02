import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime
import pickle
import logging
import sys
from sklearn import preprocessing
from datetime import datetime as dt
from keras.models import load_model
sys.path.append('./src/assets/')
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)

# constantes para extraer datos
logging.info('************* Inicio **************')
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

# variables para High
logging.info('************* Cargando Variables **************')
variablesh = pd.read_csv('./src/assets/variables/variablesHigh.csv')
variablesh = list(variablesh['0'].values)

# variables para Low
variablesl = pd.read_csv('./src/assets/variables/variablesLow.csv')
variablesl = list(variablesl['0'].values)


# carga de modelos
#logging.info('************* Cargando Modelos  Linea Regression (High)**************')
#lrH_6 = pickle.load(open('./src/assets/models/lrHigh-6.h5', 'rb'))
#lrH_5 = pickle.load(open('./src/assets/models/lrHigh-5.h5', 'rb'))
#lrH_4 = pickle.load(open('./src/assets/models/lrHigh-4.h5', 'rb'))
#lrH_3 = pickle.load(open('./src/assets/models/lrHigh-3.h5', 'rb'))
#lrH_2 = pickle.load(open('./src/assets/models/lrHigh-2.h5', 'rb'))
#lrH_1 = pickle.load(open('./src/assets/models/lrHigh-1.h5', 'rb'))
#lrH0 = pickle.load(open('./src/assets/models/lrHigh0.h5', 'rb'))
#lrH1 = pickle.load(open('./src/assets/models/lrHigh1.h5', 'rb'))
#lrH2 = pickle.load(open('./src/assets/models/lrHigh2.h5', 'rb'))
#lrH3 = pickle.load(open('./src/assets/models/lrHigh3.h5', 'rb'))
#lrH4 = pickle.load(open('./src/assets/models/lrHigh4.h5', 'rb'))
#lrH5 = pickle.load(open('./src/assets/models/lrHigh5.h5', 'rb'))
#lrH6 = pickle.load(open('./src/assets/models/lrHigh6.h5', 'rb'))

#logging.info('************* Cargando Modelos  Random Forest (High)**************')
#rfH_6 = pickle.load(open('./src/assets/models/rfHigh-6.h5', 'rb'))
#rfH_5 = pickle.load(open('./src/assets/models/rfHigh-5.h5', 'rb'))
#rfH_4 = pickle.load(open('./src/assets/models/rfHigh-4.h5', 'rb'))
#rfH_3 = pickle.load(open('./src/assets/models/rfHigh-3.h5', 'rb'))
#rfH_2 = pickle.load(open('./src/assets/models/rfHigh-2.h5', 'rb'))
#rfH_1 = pickle.load(open('./src/assets/models/rfHigh-1.h5', 'rb'))
#rfH0 = pickle.load(open('./src/assets/models/rfHigh0.h5', 'rb'))
#rfH1 = pickle.load(open('./src/assets/models/rfHigh1.h5', 'rb'))
#rfH2 = pickle.load(open('./src/assets/models/rfHigh2.h5', 'rb'))
#rfH3 = pickle.load(open('./src/assets/models/rfHigh3.h5', 'rb'))
#rfH4 = pickle.load(open('./src/assets/models/rfHigh4.h5', 'rb'))
#rfH5 = pickle.load(open('./src/assets/models/rfHigh5.h5', 'rb'))
#rfH6 = pickle.load(open('./src/assets/models/rfHigh6.h5', 'rb'))

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

#logging.info('************* Cargando Modelos  Neural Networks (High)**************')
#nnH_6 = load_model('./src/assets/models/nnHigh-6.h5')
#nnH_5 = load_model('./src/assets/models/nnHigh-5.h5')
#nnH_4 = load_model('./src/assets/models/nnHigh-4.h5')
#nnH_3 = load_model('./src/assets/models/nnHigh-3.h5')
#nnH_2 = load_model('./src/assets/models/nnHigh-2.h5')
#nnH_1 = load_model('./src/assets/models/nnHigh-1.h5')
#nnH0 = load_model('./src/assets/models/nnHigh0.h5')
#nnH1 = load_model('./src/assets/models/nnHigh1.h5')
#nnH2 = load_model('./src/assets/models/nnHigh2.h5')
#nnH3 = load_model('./src/assets/models/nnHigh3.h5')
#nnH4 = load_model('./src/assets/models/nnHigh4.h5')
#nnH5 = load_model('./src/assets/models/nnHigh5.h5')
#nnH6 = load_model('./src/assets/models/nnHigh6.h5')

#logging.info('************* Cargando Modelos  Linear Regression (Low)**************')
#lrL_6 = pickle.load(open('./src/assets/models/lrLow-6.h5', 'rb'))
#lrL_5 = pickle.load(open('./src/assets/models/lrLow-5.h5', 'rb'))
#lrL_4 = pickle.load(open('./src/assets/models/lrLow-4.h5', 'rb'))
#lrL_3 = pickle.load(open('./src/assets/models/lrLow-3.h5', 'rb'))
#lrL_2 = pickle.load(open('./src/assets/models/lrLow-2.h5', 'rb'))
#lrL_1 = pickle.load(open('./src/assets/models/lrLow-1.h5', 'rb'))
#lrL0 = pickle.load(open('./src/assets/models/lrLow0.h5', 'rb'))
#lrL1 = pickle.load(open('./src/assets/models/lrLow1.h5', 'rb'))
#lrL2 = pickle.load(open('./src/assets/models/lrLow2.h5', 'rb'))
#lrL3 = pickle.load(open('./src/assets/models/lrLow3.h5', 'rb'))
#lrL4 = pickle.load(open('./src/assets/models/lrLow4.h5', 'rb'))
#lrL5 = pickle.load(open('./src/assets/models/lrLow5.h5', 'rb'))
#lrL6 = pickle.load(open('./src/assets/models/lrLow6.h5', 'rb'))
#
#logging.info('************* Cargando Modelos  Random Forest (Low)**************')
#rfL_6 = pickle.load(open('./src/assets/models/rfLow-6.h5', 'rb'))
#rfL_5 = pickle.load(open('./src/assets/models/rfLow-5.h5', 'rb'))
#rfL_4 = pickle.load(open('./src/assets/models/rfLow-4.h5', 'rb'))
#rfL_3 = pickle.load(open('./src/assets/models/rfLow-3.h5', 'rb'))
#rfL_2 = pickle.load(open('./src/assets/models/rfLow-2.h5', 'rb'))
#rfL_1 = pickle.load(open('./src/assets/models/rfLow-1.h5', 'rb'))
#rfL0 = pickle.load(open('./src/assets/models/rfLow0.h5', 'rb'))
#rfL1 = pickle.load(open('./src/assets/models/rfLow1.h5', 'rb'))
#rfL2 = pickle.load(open('./src/assets/models/rfLow2.h5', 'rb'))
#rfL3 = pickle.load(open('./src/assets/models/rfLow3.h5', 'rb'))
#rfL4 = pickle.load(open('./src/assets/models/rfLow4.h5', 'rb'))
#rfL5 = pickle.load(open('./src/assets/models/rfLow5.h5', 'rb'))
#rfL6 = pickle.load(open('./src/assets/models/rfLow6.h5', 'rb'))

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

#logging.info('************* Cargando Modelos  Neural Networks (Low)**************')
#nnL_6 = load_model('./src/assets/models/nnLow-6.h5')
#nnL_5 = load_model('./src/assets/models/nnLow-5.h5')
#nnL_4 = load_model('./src/assets/models/nnLow-4.h5')
#nnL_3 = load_model('./src/assets/models/nnLow-3.h5')
#nnL_2 = load_model('./src/assets/models/nnLow-2.h5')
#nnL_1 = load_model('./src/assets/models/nnLow-1.h5')
#nnL0 = load_model('./src/assets/models/nnLow0.h5')
#nnL1 = load_model('./src/assets/models/nnLow1.h5')
#nnL2 = load_model('./src/assets/models/nnLow2.h5')
#nnL3 = load_model('./src/assets/models/nnLow3.h5')
#nnL4 = load_model('./src/assets/models/nnLow4.h5')
#nnL5 = load_model('./src/assets/models/nnLow5.h5')
#nnL6 = load_model('./src/assets/models/nnLow6.h5')

# descarga de datos
logging.info('************* Descargando datos **************')
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


print(sd.head())


# predicciones
logging.info('************* Haciendo Predicciones **************')
#Xh_lrH_6 = lrH_6.predict(Xh)
#Xh_lrH_5 = lrH_5.predict(Xh)
#Xh_lrH_4 = lrH_4.predict(Xh)
#Xh_lrH_3 = lrH_3.predict(Xh)
#Xh_lrH_2 = lrH_2.predict(Xh)
#Xh_lrH_1 = lrH_1.predict(Xh)
#Xh_lrH0 = lrH0.predict(Xh)
#Xh_lrH1 = lrH1.predict(Xh)
#Xh_lrH2 = lrH2.predict(Xh)
#Xh_lrH3 = lrH3.predict(Xh)
#Xh_lrH4 = lrH4.predict(Xh)
#Xh_lrH5 = lrH5.predict(Xh)
#Xh_lrH6 = lrH6.predict(Xh)

#Xh_rfH_6 = rfH_6.predict_proba(Xh)[:,1]
#Xh_rfH_5 = rfH_5.predict_proba(Xh)[:,1]
#Xh_rfH_4 = rfH_4.predict_proba(Xh)[:,1]
#Xh_rfH_3 = rfH_3.predict_proba(Xh)[:,1]
#Xh_rfH_2 = rfH_2.predict_proba(Xh)[:,1]
#Xh_rfH_1 = rfH_1.predict_proba(Xh)[:,1]
#Xh_rfH0 = rfH0.predict_proba(Xh)[:,1]
#Xh_rfH1 = rfH1.predict_proba(Xh)[:,1]
#Xh_rfH2 = rfH2.predict_proba(Xh)[:,1]
#Xh_rfH3 = rfH3.predict_proba(Xh)[:,1]
#Xh_rfH4 = rfH4.predict_proba(Xh)[:,1]
#Xh_rfH5 = rfH5.predict_proba(Xh)[:,1]
#Xh_rfH6 = rfH6.predict_proba(Xh)[:,1]

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

Xl_gbH_6 = gbL_6.predict_proba(Xl)[:,1]
Xl_gbH_5 = gbL_5.predict_proba(Xl)[:,1]
Xl_gbH_4 = gbL_4.predict_proba(Xl)[:,1]
Xl_gbH_3 = gbL_3.predict_proba(Xl)[:,1]
Xl_gbH_2 = gbL_2.predict_proba(Xl)[:,1]
Xl_gbH_1 = gbL_1.predict_proba(Xl)[:,1]
Xl_gbH0 = gbL0.predict_proba(Xl)[:,1]
Xl_gbH1 = gbL1.predict_proba(Xl)[:,1]
Xl_gbH2 = gbL2.predict_proba(Xl)[:,1]
Xl_gbH3 = gbL3.predict_proba(Xl)[:,1]
Xl_gbH4 = gbL4.predict_proba(Xl)[:,1]
Xl_gbH5 = gbL5.predict_proba(Xl)[:,1]
Xl_gbH6 = gbL6.predict_proba(Xl)[:,1]

#Xh_nnH_6 = nnH_6.predict(Xh)
#Xh_nnH_5 = nnH_5.predict(Xh)
#Xh_nnH_4 = nnH_4.predict(Xh)
#Xh_nnH_3 = nnH_3.predict(Xh)
#Xh_nnH_2 = nnH_2.predict(Xh)
#Xh_nnH_1 = nnH_1.predict(Xh)
#Xh_nnH0 = nnH0.predict(Xh)
#Xh_nnH1 = nnH1.predict(Xh)
#Xh_nnH2 = nnH2.predict(Xh)
#Xh_nnH3 = nnH3.predict(Xh)
#Xh_nnH4 = nnH4.predict(Xh)
#Xh_nnH5 = nnH5.predict(Xh)
#Xh_nnH6 = nnH6.predict(Xh)

#Xh_meanH_6 = (Xh_lrH_6 + Xh_rfH_6 + Xh_gbH_6 + Xh_nnH_6.T[0])/4
#Xh_meanH_5 = (Xh_lrH_5 + Xh_rfH_5 + Xh_gbH_5 + Xh_nnH_5.T[0])/4
#Xh_meanH_4 = (Xh_lrH_4 + Xh_rfH_4 + Xh_gbH_4 + Xh_nnH_4.T[0])/4
#Xh_meanH_3 = (Xh_lrH_3 + Xh_rfH_3 + Xh_gbH_3 + Xh_nnH_3.T[0])/4
#Xh_meanH_2 = (Xh_lrH_2 + Xh_rfH_2 + Xh_gbH_2 + Xh_nnH_2.T[0])/4
#Xh_meanH_1 = (Xh_lrH_1 + Xh_rfH_1 + Xh_gbH_1 + Xh_nnH_1.T[0])/4
#Xh_meanH0 = (Xh_lrH0 + Xh_rfH0 + Xh_gbH0 + Xh_nnH0.T[0])/4
#Xh_meanH1 = (Xh_lrH1 + Xh_rfH1 + Xh_gbH1 + Xh_nnH1.T[0])/4
#Xh_meanH2 = (Xh_lrH2 + Xh_rfH2 + Xh_gbH2 + Xh_nnH2.T[0])/4
#Xh_meanH3 = (Xh_lrH3 + Xh_rfH3 + Xh_gbH3 + Xh_nnH3.T[0])/4
#Xh_meanH4 = (Xh_lrH4 + Xh_rfH4 + Xh_gbH4 + Xh_nnH4.T[0])/4
#Xh_meanH5 = (Xh_lrH5 + Xh_rfH5 + Xh_gbH5 + Xh_nnH5.T[0])/4
#Xh_meanH6 = (Xh_lrH6 + Xh_rfH6 + Xh_gbH6 + Xh_nnH6.T[0])/4

preds = {
'Xh_gbH_6':Xh_gbH_6,
'Xh_gbH_5':Xh_gbH_5,
'Xh_gbH_4':Xh_gbH_4,
'Xh_gbH_3':Xh_gbH_3,
'Xh_gbH_2':Xh_gbH_2,
'Xh_gbH_1':Xh_gbH_1,
'Xh_gbH0':Xh_gbH0,
'Xh_gbH1':Xh_gbH1,
'Xh_gbH2':Xh_gbH2,
'Xh_gbH3':Xh_gbH3,
'Xh_gbH4':Xh_gbH4,
'Xh_gbH5':Xh_gbH5,
'Xh_gbH6':Xh_gbH6
}

logging.info(preds)
logging.info('Done')

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