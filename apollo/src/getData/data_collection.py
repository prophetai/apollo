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