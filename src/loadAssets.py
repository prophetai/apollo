import sys
import pickle
import logging
import pandas as pd
from processData.processing import setup_data, get_indicators


class Assets():

    def __init__(self, model_version, instrument):
        self.model_version = model_version
        self.instrument = instrument

    def load_vals(self):
        # variables para High
        variablesh = pd.read_csv(
            f'./assets/{self.model_version}/variables/variablesHigh.csv')
        variablesh = list(variablesh['0'].values)

        # variables para Low
        variablesl = pd.read_csv(
            f'./assets/{self.model_version}/variables/variablesLow.csv')
        variablesl = list(variablesl['0'].values)

        return variablesh, variablesl

    # carga de modelos
    def load_models(self):
        files = {}
        logging.info(
            f'************* Cargando Modelos ({self.model_version}) Gradient Boosting (High)**************')
        files['gbHigh-6']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-6.h5', 'rb')))
        files['gbHigh-5']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-5.h5', 'rb')))
        files['gbHigh-4']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-4.h5', 'rb')))
        files['gbHigh-3']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-3.h5', 'rb')))
        files['gbHigh-2']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-2.h5', 'rb')))
        files['gbHigh-1']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh-1.h5', 'rb')))
        files['gbHigh0']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh0.h5', 'rb')))
        files['gbHigh1']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh1.h5', 'rb')))
        files['gbHigh2']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh2.h5', 'rb')))
        files['gbHigh3']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh3.h5', 'rb')))
        files['gbHigh4']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh4.h5', 'rb')))
        files['gbHigh5']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh5.h5', 'rb')))
        files['gbHigh6']=(pickle.load(
            open(f'./assets/{self.model_version}/gbHigh6.h5', 'rb')))

        logging.info(
            f'************* Cargando Modelos ({self.model_version}) Gradient Boosting (Low)**************')
        files['gbLow-6']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-6.h5', 'rb')))
        files['gbLow-5']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-5.h5', 'rb')))
        files['gbLow-4']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-4.h5', 'rb')))
        files['gbLow-3']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-3.h5', 'rb')))
        files['gbLow-2']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-2.h5', 'rb')))
        files['gbLow-1']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow-1.h5', 'rb')))
        files['gbLow0']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow0.h5', 'rb')))
        files['gbLow1']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow1.h5', 'rb')))
        files['gbLow2']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow2.h5', 'rb')))
        files['gbLow3']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow3.h5', 'rb')))
        files['gbLow4']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow4.h5', 'rb')))
        files['gbLow5']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow5.h5', 'rb')))
        files['gbLow6']=(pickle.load(
            open(f'./assets/{self.model_version}/gbLow6.h5', 'rb')))

        return files

    def load_scaler(self):
        scaler_high = pickle.load(
            open(f'./assets/{self.model_version}/scalerH', 'rb'))
        scaler_low = pickle.load(
            open(f'./assets/{self.model_version}/scalerL', 'rb'))

        return scaler_high, scaler_low
