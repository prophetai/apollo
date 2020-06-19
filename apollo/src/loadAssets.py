import pickle
import logging
import pandas as pd
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators


class Assets():

    def __init__(self, model_version, instrument):
            self.model_version = model_version
            self.instrument = instrument

    def load_vals(self):
        # variables para High
        logging.info('************* Cargando Variables **************')
        variablesh = pd.read_csv(
            f'./src/assets/{self.model_version}/variablesHigh.csv')
        variablesh = list(variablesh['0'].values)

        # variables para Low
        variablesl = pd.read_csv(
            f'./src/assets/{self.model_version}/variablesLow.csv')
        variablesl = list(variablesl['0'].values)

        return variablesh, variablesl

    # carga de modelos
    def load_models(self):
        files = []
        logging.info(
            f'************* Cargando Modelos {self.model_version}Gradient Boosting (High)**************')

        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-6.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-5.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-4.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-3.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-2.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh-1.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh0.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh1.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh2.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh3.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh4.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh5.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbHigh6.h5', 'rb')))

        logging.info(
            '************* Cargando Modelos  Gradient Boosting (Low)**************')
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-6.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-5.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-4.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-3.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-2.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow-1.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow0.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow1.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow2.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow3.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow4.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow5.h5', 'rb')))
        files.append(pickle.load(
            open(f'./src/assets/{self.model_version}/gbLow6.h5', 'rb')))

        models = {}
        
        models = {f'Xh_gbH-{i}': file_model.predict_proba(Xh) for i,file_model in enumerate(files[:len(files)/2])}

        
        models['Xh_gbH-6'] = gbH_6.predict_proba(Xh)[:, 1]
        models['Xh_gbH-5'] = gbH_5.predict_proba(Xh)[:, 1]
        models['Xh_gbH-4'] = gbH_4.predict_proba(Xh)[:, 1]
        models['Xh_gbH-3'] = gbH_3.predict_proba(Xh)[:, 1]
        models['Xh_gbH-2'] = gbH_2.predict_proba(Xh)[:, 1]
        models['Xh_gbH-1'] = gbH_1.predict_proba(Xh)[:, 1]
        models['Xh_gbH0'] = gbH0.predict_proba(Xh)[:, 1]
        models['Xh_gbH1'] = gbH1.predict_proba(Xh)[:, 1]
        models['Xh_gbH2'] = gbH2.predict_proba(Xh)[:, 1]
        models['Xh_gbH3'] = gbH3.predict_proba(Xh)[:, 1]
        models['Xh_gbH4'] = gbH4.predict_proba(Xh)[:, 1]
        models['Xh_gbH5'] = gbH5.predict_proba(Xh)[:, 1]
        models['Xh_gbH6'] = gbH6.predict_proba(Xh)[:, 1]

        models['Xl_gbl-6'] = gbL_6.predict_proba(Xl)[:, 1]
        models['Xl_gbl-5'] = gbL_5.predict_proba(Xl)[:, 1]
        models['Xl_gbl-4'] = gbL_4.predict_proba(Xl)[:, 1]
        models['Xl_gbl-3'] = gbL_3.predict_proba(Xl)[:, 1]
        models['Xl_gbl-2'] = gbL_2.predict_proba(Xl)[:, 1]
        models['Xl_gbl-1'] = gbL_1.predict_proba(Xl)[:, 1]
        models['Xl_gbl0'] = gbL0.predict_proba(Xl)[:, 1]
        models['Xl_gbl1'] = gbL1.predict_proba(Xl)[:, 1]
        models['Xl_gbl2'] = gbL2.predict_proba(Xl)[:, 1]
        models['Xl_gbl3'] = gbL3.predict_proba(Xl)[:, 1]
        models['Xl_gbl4'] = gbL4.predict_proba(Xl)[:, 1]
        models['Xl_gbl5'] = gbL5.predict_proba(Xl)[:, 1]
        models['Xl_gbl6'] = gbL6.predict_proba(Xl)[:, 1]

        return files

    def load_scaler(self):
        scaler_high = pickle.load(open(f'./src/assets/{self.model_version}/scalerH', 'rb'))
        scaler_low = pickle.load(open(f'./src/assets/{self.model_version}/scalerL', 'rb'))

        return scaler_high, scaler_low
