import logging
import pandas as pd
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators
from loadAssets import Assets

class Data():
    def __init__(self,**setup):
        '''
        Instance of Data class. Sets up for getting data

        Input:
            - **setup(dict):
                - instrument:(str) trading instrument
                - ins_variables:(str) investement variables
                - granularity:(str) granularity of the data (H1,D1,etc)
                - start:(datetime) Date start
                - end:(datetime) Date end
                - freq:(str) frequency (daily, hourly, etc [D,H...])
                - candleformat:(str) (data format)
                - trading: (boolean.optional) True if it is used for live trading
        '''
        self.instrument = setup['instrument']
        self.ins_variables = setup['ins_variables']
        self.granularity = setup['granularity'] #'H1'
        self.start = setup['start'] #str(dt.now() + timedelta(days=-2))[:10]
        self.end = setup['end'] #str(dt.now())[:10]
        self.freq = setup['freq'] #'D'
        self.data_set = None
        self.candleformat = setup['candleformat']
        self.trading = setup['trading'] or False

    def get_dataset(self):
        gf = get_forex(self.instrument,
                       self.ins_variables,
                       self.granularity,
                       self.start,
                       self.end,
                       self.candleformat,
                       self.freq,
                       self.trading)
        
        data = setup_data(gf,
                instrument=self.instrument,
                pricediff=True,
                log=True,
                trading=self.trading)
        
        return data,gf

    def process_data(self, data):
        processeddf = get_indicators(data, 
                             self.instrument, 
                             column='{}_closeBid'.format(self.instrument), 
                             wind=10, 
                             bidask='Bid') 
        processeddf = get_indicators(processeddf, 
                             self.instrument, 
                             column='{}_closeAsk'.format(self.instrument), 
                             wind=10, 
                             bidask='Ask')

        processeddf = processeddf.fillna(method='ffill')
        processeddf = processeddf.fillna(method='bfill')

        return processeddf

    def get_data(self,model_version):
        
        # carga escaladores
        assets = Assets(model_version,self.instrument)
        scaler_high, scaler_low = assets.load_scaler()
        # carga variables
        
        logging.info('************* Cargando Variables **************')
        variablesh, variablesl = assets.load_vals()                
        logging.info('************* Obteniendo Datos **************')
        data, original_dataset = self.get_dataset()
        logging.info('************* Procesando Datos **************')
        data = self.process_data(data)
        
        logging.info('************* Escalando Datos **************')
        Xh = data[variablesh]
        Xh = scaler_high.transform(Xh)

        Xl = data[variablesl]
        Xl = scaler_low.transform(Xl)

        return Xh, Xl, original_dataset