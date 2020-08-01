import sys
import pickle
import logging
import pandas as pd
from getData.extract import get_forex
from processData.processing import setup_data, get_indicators
# sys.path.append('./src/assets/')
# sys.path.append('./src')


class Assets():

    def __init__(self, model_version, instrument):
        self.model_version = model_version
        self.instrument = instrument

    def load_vals(self):
        # variables para High
        variablesh = pd.read_csv(
            f'./src/assets/{self.model_version}/variables/variablesHigh.csv')
        variablesh = list(variablesh['0'].values)

        # variables para Low
        variablesl = pd.read_csv(
            f'src/assets/{self.model_version}/variables/variablesLow.csv')
        variablesl = list(variablesl['0'].values)

        return variablesh, variablesl

    # carga de modelos
    def load_models(self):
        files = []
        logging.info(
            f'************* Cargando Modelos ({self.model_version}) Gradient Boosting (High)**************')

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
            f'************* Cargando Modelos ({self.model_version}) Gradient Boosting (Low)**************')
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

        return files

    def load_scaler(self):
        scaler_high = pickle.load(
            open(f'./src/assets/{self.model_version}/scalerH', 'rb'))
        scaler_low = pickle.load(
            open(f'./src/assets/{self.model_version}/scalerL', 'rb'))

        return scaler_high, scaler_low
