import pandas as pd
import numpy as np
import oandapy as opy
import psycopg2
import logging
import os
from datetime import datetime as dt
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)

def get_forex(instrument,
              instruments,
              granularity,
              start,
              end,
              candleformat,
              freq,
              trading=False):
    """
    Obtiene datos FX de Oanda para los instrumentos que elijamos

    Args:
        instrument (str): Instrumento a predecir
        instruments (list): Divisas
        granularity (str): Time Window
        start (str): Primer día
        end (str): último día
        candleformat (str): 'bidask' o 'midpoint'
        freq (str): Timeframe
        trading (bool): Si estamos en producción
    Returns:
        df (DataFrame)
    """

    oanda = opy.API(environment='live')
    divs = {}

    for j in instruments:
        logging.info(j)
        # Extraemos datos cada 2 días (por simplicidad)
        d1 = start
        d2 = end
        dates = pd.date_range(start=d1, end=d2, freq=freq)
        df = pd.DataFrame()

        if trading:
            data = oanda.get_history(instrument=j,
                                         candleFormat=candleformat,
                                         since=d1,
                                         granularity=granularity)
            df = pd.DataFrame(data['candles'])
        else:
            pbar = tqdm(total=len(dates) - 1)

            for i in range(0, len(dates) - 1):
                # Oanda toma las fechas en este formato
                d1 = str(dates[i]).replace(' ', 'T')
                d2 = str(dates[i+1]).replace(' ', 'T')
                try:
                    # Concatenamos cada día en el dataframe

                    data = oanda.get_history(instrument=j,
                                             candleFormat=candleformat,
                                             start=d1,
                                             end=d2,
                                             granularity=granularity)

                    df = df.append(pd.DataFrame(data['candles']))
                    pbar.update(1)
                except:
                    pass

        if trading == False:
            pbar.close()
        date = pd.DatetimeIndex(df['time'], tz='UTC')
        df['date'] = date
        cols = [j + '_' + k for k in df.columns]
        df.columns = cols
        divs[j] = df

    dat = divs[instruments[0]]
    for i in instruments[1:]:
        join_id = [k for k in divs[i].columns if 'date' in k][0]
        dat = pd.merge(dat,
                       divs[i],
                       left_on=instrument + '_date',
                       right_on=join_id, how='left')

    return dat

def calculate_difference(dat, log=True, drop=False):
    """
    Calcula diferencia entre el valor actual y el pasado

    Args:
        df (DataFrame): Datos
        log (bool): Diferencia logaritmica
    Returns:
        df (DataFrame): Datos con diferencias
    """
    df = dat.copy()
    drop_vars = []
    for i in df.columns:
        try:
            if log:
                df['LogDiff ' + i] = np.log(df[i]).diff(1)
            else:
                df['Diff ' + i] = df[i].diff(1)
            drop_vars.append(i)

        except Exception as e:
            logging.warning(e)

    if drop:
        df = df.drop(drop_vars, axis=1)

    return df, drop_vars

def setup_data(dat,
               instrument='USD_JPY',
               pricediff=True,
               log=True,
               trading=False):
    """
    Calcula diferencias y acomoda datos

    Args:
        dat (DataFrame): Datos
        instrument (str): Divisa
        pricediff (bool): Si queremos diferencias en precios
        log (bool): Si queremos transformación logarítmica
        trading (bool): Si estamos en producción
    Returns:
        df (DataFrame): Datos transformados y ajustados
    """

    df = dat.copy()
    date = '{}_date'.format(instrument)
    drops = [k for k in df.columns if date not in k and ('date' in k or 'complete' in k or 'time' in k)]
    df = df.drop(drops, axis=1)
    if trading == False:
        df = df[100:] # Falla en API
    df = df.reset_index(drop=True)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    drops = []
    if pricediff:
        df, drops = calculate_difference(df, log=log)

    high = instrument + '_highMid'
    low = instrument + '_lowMid'
    close = instrument + '_closeMid'
    hcdiff = 'Diff High-Close'
    cldiff = 'Diff Close-Low'
    hldiff = 'Diff High-Low'

    df[hcdiff] = df[high] - df[close]
    df[cldiff] = df[close] - df[low]
    df[hldiff] = df[high] - df[low]

    drops = [i for i in drops if i not in [date, hcdiff, cldiff, hldiff] and 'volume' not in i]
    #df = df.drop(drops, axis=1)
    df = df[1:]
    df[date] = df[date].astype(str)
    df[date] = df[date].str[:13]
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df

from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif(X):
    """
    Nos da el factor de inflación de la varianza de cada variable independiente

    Args:
        X (DataFrame): DataFrame con datos de nuestras variables independientes
    Returns:
        vif (DataFrame): DataFrame con el factor de inflación de la varianza
                         de cada variable
    """
    vif = pd.DataFrame()
    X['intercept'] = 1
    x = X.values
    vif['vif'] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    vif['feature'] = X.columns

    return vif

def reduce_multicol_randomly(data, instrument, dontdrop=[]):
    """
    Reduce multicolinealidad aleatoriamente

    Args:
        data (DataFrame): Datos
        instrument (str): Divisa
        dontdrop (list): Variables que no queremos quitar

    Returns:
        df (DataFrame): Datos con multicolinealidad reducida
    """

    df = data.copy()
    dd = [i for i in df.columns if instrument not in i and 'Mid' not in i]
    dontdrop += dd
    dropping = [1, 2]
    dat = df.drop(dontdrop, axis=1)
    while len(dropping) >= 2:
        vif = get_vif(dat)
        svif = vif.sort_values('vif').reset_index(drop=True)
        display(svif)
        dropping = svif[svif['vif'] >= 100]
        try:
            vif_drops = list(dropping.sample(n=int(len(dropping)/2))['feature'].values)
            dat = dat.drop(vif_drops, axis=1)
        except:
            display(svif)

    df['intercept'] = 1
    dfcols = list(dat.columns) + dontdrop
    df = df[dfcols]

    return df

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math

def train_test(df, response, train_size=0.75, time_series=False, scaling=None):
    """
    Regresa train y test sets

    Args:
        df (DataFrame): Datos listos para el modelo
        response (str): Variable respuesta
        train_size (float): % Train Size
        time_series (boolean): Si es serie de tiempo o no
        scaling (str): ['standard', 'minmax', 'maxabs', 'robust', 'quantile']
    Returns:
        X_train (Array): conjunto de datos de entrenamiento (indep)
        X_test (Array): conjunto de datos de prueba (indep)
        y_train (Array): conjunto de datos de entrenamiento (dep)
        y_test (Array): conjunto de datos de prueba (dep)
    """

    data = df.copy()
    X = data.drop(response, 1)
    y = data[response]

    logging.info('X columns')
    logging.info(list(X.columns))
    logging.info('Response')
    logging.info(response)

    if time_series:
        trainsize = int(train_size*len(X))
        X_train = X[:trainsize].values
        X_test = X[trainsize:].values
        y_train = y[:trainsize].values
        y_test = y[trainsize:].values

    else:
        X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                            y.values,
                                                            random_state=0,
                                                            train_size=train_size)
    if scaling == 'standard':
        scaler = preprocessing.StandardScaler()
    if scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    if scaling == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    if scaling == 'robust':
        scaler = preprocessing.RobustScaler()
    if scaling == 'quantile':
        scaler = preprocessing.QuantileTransformer()

    if scaling != None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def logreg(X_train, y_train):
    """
    Calcula modelo de Regresión Logística
    Args:
        X_train (Array): conjunto de datos de entrenamiento (regresores)
        y_train (Array): conjunto de datos de entrenamiento (objetivo)
    Returns:
        logreg (modelo): Regresión Logística
    """
    try:
        # Si la matriz es singular va a dar error
        log = sm.Logit(y_train, X_train)
        logreg_model = log.fit()
    except Exception as e:
        # Intentamos con la matriz hessiana
        logging.error(e)
        log = sm.Logit(y_train, X_train)
        logreg_model = log.fit(method='bfgs')

    return logreg_model

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


def model_precision(y, predictions, sc, disp=False):
    """
    Métricas de precisión de un modelo de clasificación

    Args:
        y (array): Instancias de la variable dependiente
        predictions (array): Predicciones
        sc (float): Score de corte entre 0 y 1 que marca el límite de clasificación
                    (arriba de sc se considera positivo)
        disp (boolean): Imprimir matriz con métricas
    Returns:
        accuracy (float): (tp+tn)/(tp+tn+fp+fn)
        precision (float): tp/(tp+fp)
        recall (float): tp/(tp+fn)
        f1_score (float): 2/(1/Precision+1/Recall) Media armónica
                          entre Precision y Recall
        mcc (float): Matthiews Correlation Coefficient
                     (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

    """
    yt = y.copy()
    predictionst = predictions.copy()

    yt = yt.reshape([yt.shape[0], 1])
    predictionst = predictionst.reshape([predictionst.shape[0], 1])

    test = np.concatenate((yt, predictionst),axis=1)

    tp = ((test[:,0] == 1) & (test[:,1] >= sc)).sum()
    fp = ((test[:,0] == 0) & (test[:,1] >= sc)).sum()
    tn = ((test[:,0] == 0) & (test[:,1] < sc)).sum()
    fn = ((test[:,0] == 1) & (test[:,1] < sc)).sum()

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2/(1/precision+1/recall)
    mcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

    res = pd.DataFrame(0, index=['Accuracy', 'Precision',
                                 'Recall', 'F1 Score',
                                 'MCC'], columns=['Score'])

    res.loc['Accuracy'] = accuracy
    res.loc['Precision'] = precision
    res.loc['Recall'] = recall
    res.loc['F1 Score'] = f1_score
    res.loc['MCC'] = mcc

    if disp != False:
        display(res)

    return accuracy, precision, recall, f1_score, mcc

def tenbin_cutscore(y, predictions):
    """
    Precision por intervalos de 10 en 10

    Args:
        y (array): Instancias de la variable dependiente
        predictions (array): Predicciones

    Returns:
        res (DataFrame): Métricas de precisión por intervalos

    """
    yt = y.copy()
    predictionst = predictions.copy()

    scoresindex = ['0-10',
                   '10-20',
                   '20-30',
                   '30-40',
                   '40-50',
                   '50-60',
                   '60-70',
                   '70-80',
                   '80-90',
                   '90-100']
    scorescolumns = ['Total','Positives']
    res = pd.DataFrame(0, index=scoresindex, columns=scorescolumns)

    yt.shape = [yt.shape[0], 1]
    predictionst.shape = [predictionst.shape[0], 1]

    test = np.concatenate((yt, predictionst), axis=1)

    low = 0
    up = 0.1
    for i in scoresindex:
        res.loc[i]['Total'] = ((test[:,1] >= low) & (test[:,1] < up)).sum()
        res.loc[i]['Positives'] = ((test[:,1] >= low) & (test[:,1] < up) & \
                                   (test[:,0] == 1)).sum()
        low += 0.1
        up += 0.1
    res['Positive Rate'] = res['Positives'].div(res['Total'])*100
    res['% of Total'] = res['Total'].div(res['Total'].sum())*100
    res['% of Positives'] = res['Positives'].div(res['Positives'].sum())*100

    return res

def qcut_precision(y, predictions, n):
    """
    Precision por cuantiles

    Args:
        y (array): Instancias de la variable dependiente
        predictions (array): Predicciones
        n (int): Número de cuantiles

    Returns:
        res (DataFrame): Métricas de precisión por cuantiles

    """
    predictionst = predictions.copy()
    predictionst.shape = (predictionst.shape[0], )
    ylist = y.tolist()
    plist = predictionst.tolist()

    cuts = pd.DataFrame({'y':ylist, 'predictions':plist})
    qcuts = pd.qcut(cuts['predictions'], n, duplicates='drop')
    cuts['qcut'] = qcuts
    cuts['qcut'] = cuts['qcut'].astype(str)

    ones = []
    zeros = []
    mean_score = []
    total = []
    for i in cuts['qcut'].unique():
        ones.append(len(cuts[(cuts['qcut'] == i) & (cuts['y'] != 0)]))
        zeros.append(len(cuts[(cuts['qcut'] == i) & (cuts['y'] != 1)]))
        total.append(len(cuts[cuts['qcut'] == i]))
        mean_score.append(cuts[cuts['qcut'] == i]['predictions'].mean())

    res = pd.DataFrame({'Total': total,
                        'Positives': ones,
                        '#0s': zeros,
                        'Mean Score': mean_score})

    res['Positive Rate'] = res['Positives'].div(res['Total'])*100

    tp = []
    fp = []
    tn = []
    fn = []
    for i in res['Mean Score'].unique():
        tp.append(len(cuts[(cuts['predictions'] >= i) & (cuts['y'] != 0)]))
        fp.append(len(cuts[(cuts['predictions'] >= i) & (cuts['y'] != 1)]))
        tn.append(len(cuts[(cuts['predictions'] < i) & (cuts['y'] != 1)]))
        fn.append(len(cuts[(cuts['predictions'] < i) & (cuts['y'] != 0)]))

    res['TP'] = tp
    res['FP'] = fp
    res['TN'] = tn
    res['FN'] = fn

    res['Accuracy'] = (res['TP'] + res['TN']).div(res['TP'] + res['TN'] + \
                                                  res['FP'] + res['FN'])
    res['Precision'] = res['TP'].div(res['TP'] + res['FP'])
    res['Recall'] = res['TP'].div(res['TP'] + res['FN'])
    res['F1-score'] = 2/(1/res['Precision'] + 1/res['Recall'])
    res['False Positive Rate'] = res['FP'].div(res['FP'] + res['TN'])

    res['% of Total'] = res['Total'].div(res['Total'].sum())*100
    res['% of Positives'] = res['Positives'].div(res['Positives'].sum())*100

    res = res[['Mean Score',
               'Total',
               'Positives',
               'Positive Rate',
               '% of Total',
               '% of Positives',
               'Accuracy',
               'Precision',
               'Recall',
               'False Positive Rate']]
    res.index += 1

    return res

def plot_roc(y, predictions):
    """
    Gráfica de la curva de roc del modelo

    Args:
        y (array): Vector de variable objetivo
        predictions (array): Vector de predicciones
    """
    fpr, tpr, thresholds = roc_curve(y,
                                     predictions,
                                     pos_label=None,
                                     sample_weight=None,
                                     drop_intermediate=True)
    plt.subplot(121)
    plt.plot(fpr, tpr)
    auc = np.trapz(tpr, fpr).round(5)
    plt.axis([0,1,0,1])
    plt.plot([0,1],[0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.text(0.65, 0.02, 'AUC: ' + str(auc), fontsize=12)
    plt.title('ROC Curve')

def plot_prc(y, predictions):
    """
    Gráfica de la curva de precision-recall del modelo

    Args:
        y (array): Vector de variable objetivo
        predictions (array): Vector de predicciones
    """
    average_precision = average_precision_score(y, predictions).round(5)
    precision, recall, threshold = precision_recall_curve(y, predictions)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.subplot(122)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.text(0.65, 0.02, 'AP: ' + str(average_precision), fontsize=12)
    plt.title('Precision-Recall Curve')


def show_metrics(y, predictions, sc=0.5, disp=True, n=10):
    """
    Muestra precisión por cuantiles, precisión por intervalos
    de score y gráficas de PRC y ROC

    Args:
        y (array): Instancias de la variable dependiente
        predictions (array): Predicciones
        sc (float): Score de corte entre 0 y 1 que marca el límite de clasificación
                    (arriba de sc se considera positivo)
        disp (boolean): Imprimir matriz con métricas
        n (int): Número de cuantiles
    """
    yt = y.copy()
    predictionst = predictions.copy()
    display(qcut_precision(yt, predictionst, n))
    display(tenbin_cutscore(yt, predictionst))
    metrics = model_precision(yt, predictionst, sc, disp)
    plot_roc(yt, predictionst)
    plot_prc(yt, predictionst)
    plt.subplots_adjust(left=3.1, right=5.1, bottom=2, top=3, hspace=0.2, wspace=0.5)
    plt.show()

def model_creation_hhll(dat, instrument, prints, scaling):
    """
    Crea modelos para pronosticar

    Args:
        dat (DataFrame): Datos para modelo
        instrument (str): Divisa
        pricediff (bool): Si queremos diferencia en precios
        prints (int): Cuántos datos imprimir en el plot
        scaling (str): Estandarización

    Returns:
        models (dict): Diccionario con modelos
        variables (dict): Diccionario con variables para cada modelo
    """
    df = dat.copy()
    DF = df.copy()

    Actuals = ['LogDiff {}_highMid'.format(instrument),
               'LogDiff {}_lowMid'.format(instrument)]

    Responses = ['future diff high',
                 'future diff low']

    models = {}
    variables = {}

    for k in [0,1,2,3,4,5,6]:
        for actual,response in zip(Actuals, Responses):
            print(response)
            print(str(k/100) + '%')
            df = DF.copy()
            if 'low' in response:
                df[response] = np.exp(df[actual].shift(-1)).apply(lambda x: 1 if x<=(1 - k/10000) else 0)
            else:
                df[response] = np.exp(df[actual].shift(-1)).apply(lambda x: 1 if x>=(1 + k/10000) else 0)
            df = df.drop(Actuals, axis=1)
            df = df.dropna()
            display(df.head())
            display(df.corr()[[response]].sort_values(response))
            X_train, X_test, y_train, y_test = train_test(df,
                                                          response,
                                                          train_size=0.75,
                                                          time_series=True,
                                                          scaling=scaling)

            X_train = sm.add_constant(X_train, prepend=True, has_constant='skip')
            X_test = sm.add_constant(X_test, prepend=True, has_constant='skip')

            lr = logreg(X_train, y_train)
            #nn = best_nn(X_train, y_train, X_test, y_test, 10, 15, 'auc', threshold=0.5)
            models["HHLL_{}".format(actual) + str(k)] = lr
            #models["NeuralNet_{}".format(actual)] = nn
            variables[actual + str(k)] = list(df.drop(response, axis=1).columns)
            print(actual)
            plt.figure(figsize=(9,4))
            predplot = lr.predict(X_test[:prints])
            predplot[predplot >= 0.5] = 0.9
            predplot[predplot < 0.5] = 0.1
            plt.plot(range(len(y_test[:prints])),
                     predplot,
                     color='r',
                     marker='o',
                     label='Predicted')
            plt.plot(range(len(y_test[:prints])),
                     y_test[:prints],
                     color='b',
                     marker='o',
                     label='Expected')
            plt.legend()
            plt.show()
            show_metrics(y_test, lr.predict(X_test))

    return models, variables

def model_creation_hcl(dat, instrument, prints, scaling):
    """
    Crea modelos para pronosticar

    Args:
        dat (DataFrame): Datos para modelo
        instrument (str): Divisa
        prints (int): Cuántos datos imprimir en el plot
        scaling (str): Estandarización

    Returns:
        models (dict): Diccionario con modelos
        variables (dict): Diccionario con variables para cada modelo
    """
    df = dat.copy()
    DF = df.copy()

    Actuals = ['Diff High-Close'.format(instrument),
               'Diff Close-Low'.format(instrument)]
    Responses = ['future high-close',
                 'future close-low']
    models = {}
    variables = {}
    for k in [1.5, 2, 2.5]:
        for actual,response in zip(Actuals, Responses):
            logging.info(k)
            df = DF.copy()
            df[response] = df[actual].shift(-1)
            display(df[[actual,response]].head())
            df[response] = df[response].div(df[actual])
            display(df[[actual,response]].head())
            df[response] = df[response].apply(lambda x: 1 if x>=k else 0)
            #df = df.drop(Actuals, axis=1)
            #df = get_bestvars(df, response, 0.05, dontdrop=None, fecha=None)
            df = df.dropna()
            display(df.corr()[[response]].sort_values(response))
            X_train, X_test, y_train, y_test = train_test(df,
                                                          response,
                                                          train_size=0.75,
                                                          time_series=True,
                                                          scaling=scaling)

            X_train = sm.add_constant(X_train, prepend=True, has_constant='skip')
            X_test = sm.add_constant(X_test, prepend=True, has_constant='skip')

            lr = logreg(X_train, y_train)
            #nn = best_nn(X_train, y_train, X_test, y_test, 10, 15, 'auc', threshold=0.5)
            models["HCL_{}".format(actual) + str(k)] = lr
            #models["NeuralNet_{}".format(actual)] = nn
            variables[actual + str(k)] = list(df.drop(response, axis=1).columns)
            logging.info(actual)
            logging.info('LOGREG')
            plt.figure(figsize=(9,4))
            predplot = lr.predict(X_test[:prints])
            predplot[predplot >= 0.5] = 0.9
            predplot[predplot < 0.5] = 0.1
            plt.plot(range(len(y_test[:prints])),
                     predplot,
                     color='r',
                     marker='o',
                     label='Predicted')
            plt.plot(range(len(y_test[:prints])),
                     y_test[:prints],
                     color='b',
                     marker='o',
                     label='Expected')
            plt.legend()
            plt.show()
            show_metrics(y_test, lr.predict(X_test))

    return models, variables

def model_creation_mc(dat, instrument, prints, scaling):
    """
    Crea modelos para pronosticar

    Args:
        dat (DataFrame): Datos para modelo
        instrument (str): Divisa
        prints (int): Cuántos datos imprimir en el plot
        scaling (str): Estandarización

    Returns:
        models (dict): Diccionario con modelos
        variables (dict): Diccionario con variables para cada modelo
    """
    df = dat.copy()
    DF = df.copy()

    actual = '{}_closeMid'.format(instrument)
    future = 'future_close'
    response = 'change'
    models = {}
    variables = {}
    df = DF.copy()
    df[future] = df[actual].shift(-1)
    df['past close'] = df[actual].shift(1)
    df['past movement'] = df[actual] - df['past close']
    df['past movement'] = df['past movement'].apply(lambda x: 1 if x>=0 else 0)
    df['movement'] = df[future] - df[actual]
    df['movement'] = df['movement'].apply(lambda x: 1 if x>=0 else 0)
    df[response] = df['past movement'] + df['movement']
    df[response] = df[response].apply(lambda x: 1 if x==1 else 0)
    #df = df.drop(['past movement'], axis=1)
    #df = df.drop(Actuals, axis=1)
    #df = get_bestvars(df, response, 0.05, dontdrop=None, fecha=None)
    df = df.dropna()
    display(df.corr()[[response]].sort_values(response))
    X_train, X_test, y_train, y_test = train_test(df,
                                                  response,
                                                  train_size=0.75,
                                                  time_series=True,
                                                  scaling=scaling)

    X_train = sm.add_constant(X_train, prepend=True, has_constant='skip')
    X_test = sm.add_constant(X_test, prepend=True, has_constant='skip')

    lr = logreg(X_train, y_train)
    #nn = best_nn(X_train, y_train, X_test, y_test, 10, 15, 'auc', threshold=0.5)
    models["MC_{}".format(actual)] = lr
    #models["NeuralNet_{}".format(actual)] = nn
    variables[actual] = list(df.drop(response, axis=1).columns)
    logging.info(actual)
    logging.info('LOGREG')
    plt.figure(figsize=(9,4))
    predplot = lr.predict(X_test[:prints])
    predplot[predplot >= 0.5] = 0.9
    predplot[predplot < 0.5] = 0.1
    plt.plot(range(len(y_test[:prints])),
             predplot,
             color='r',
             marker='o',
             label='Predicted')
    plt.plot(range(len(y_test[:prints])),
             y_test[:prints],
             color='b',
             marker='o',
             label='Expected')
    plt.legend()
    plt.show()
    show_metrics(y_test, lr.predict(X_test))

    return models, variables
