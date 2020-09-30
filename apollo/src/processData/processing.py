import pandas as pd
import numpy as np
import logging

def datatypes(df):
    """
    Generates list of variables assigning its corresponding datatype
    Args:
        df (DataFrame): DataFrame
    Returns:
        numeric (list): Numeric variables
        categoric (list): Categorical variables
        dates (list): Temporal variables (dates)
    """
    numeric = list(df.select_dtypes(include=['int', 'float', 'uint8']).columns)
    categoric = list(df.select_dtypes(include=['category', 'object']).columns)
    dates = list(df.select_dtypes(include=['datetime',
                                            'datetime64[ns]']).columns)

    return numeric, categoric, dates

def calculate_difference(dat, log=True):
    """
    Difference between actual and past value

    Args:
        dat (DataFrame): Data
        log (bool): Log difference
    Returns:
        df (DataFrame): Data
    """
    df = dat.copy()
    for i in df.columns:
        try:
            if log:
                df['LogDiff ' + i] = np.log(df[i]).diff(1)
            else:
                df['Diff ' + i] = df[i].diff(1)

        except Exception as e:
            logging.warning(f'({i}):{e}')
            

    return df

def set_window(data, window, instrument):
    """
    Args:
        data (DataFrame)
        window (int):
    """

    df = data.copy()
    times = int(60/window)
    shifts = range(1, times)

    for i in df.columns:
        for s in shifts:
            try:
                df[i + str(s*window)] = df[i].shift(-s)
            except:
                continue

    df['date'] = df['{}_date'.format(instrument)].astype(str)
    df['date'] = df['date'].str[16:]
    #df = df[df['date'] == '00:00+00:00']
    df = df[:-4]

    lows = [i for i in df.columns if 'low' in i and '{}_'.format(instrument) in i]
    highs = [i for i in df.columns if 'high' in i and '{}_'.format(instrument) in i]
    df['LOW'] = df[lows].min(axis=1)
    df['HIGH'] = df[highs].max(axis=1)

    for i,j in zip(lows, highs):
        df.loc[df[i] == df['LOW'], 'low_in_minute'] = i[-2:]
        df.loc[df[j] == df['HIGH'], 'high_in_minute'] = j[-2:]

    df.loc[df['low_in_minute'].str.contains('d'), 'low_in_minute'] = \
                            df['low_in_minute'].str[1:]
    df.loc[df['high_in_minute'].str.contains('d'), 'high_in_minute'] = \
                            df['high_in_minute'].str[1:]

    df.loc[df['low_in_minute'] == 'id', 'low_in_minute'] = 0
    df.loc[df['high_in_minute'] == 'id', 'high_in_minute'] = 0

    df['low_in_minute'] = pd.to_numeric(df['low_in_minute'])
    df['high_in_minute'] = pd.to_numeric(df['high_in_minute'])

    df['first_movement_high'] = 0
    df.loc[df['high_in_minute'] < df['low_in_minute'], 'first_movement_high'] = 1
    df = df[df['high_in_minute'] != df['low_in_minute']]
    df = df.drop(['high_in_minute', 'low_in_minute'], axis=1)

    return df

def setup_data(dat,
               instrument='USD_JPY',
               pricediff=True,
               log=True,
               trading=False):
    """
    Calculates difference and arranges data

    Args:
        dat (DataFrame): Data
        instrument (str): Objective instrument
        pricediff (bool): Price difference
        log (bool): Log transformation
        trading (bool): If trading
    Returns:
        df (DataFrame): Adjusted data
    """

    df = dat.copy()
    date = '{}_date'.format(instrument)
    drops = [k for k in df.columns if date not in k and \
            ('date' in k or 'complete' in k or 'time' in k)]
    df = df.drop(drops, axis=1)
    if trading == False:
        df = df[100:]
    df = df.reset_index(drop=True)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    drops = []
    if pricediff:
        df = calculate_difference(df, log=log)

    high_ask = instrument + '_highAsk'
    low_ask = instrument + '_lowAsk'
    close_ask = instrument + '_closeAsk'
    hcdiff_ask = 'Diff High-Close_ask'
    cldiff_ask = 'Diff Close-Low_ask'
    hldiff_ask = 'Diff High-Low_ask'
    hpcdiff_ask = 'Diff High-PastClose_ask'
    cpldiff_ask = 'Diff PastClose-Low_ask'
    hpldiff_ask = 'Diff PastHigh-Low_ask'

    df[hcdiff_ask] = df[high_ask] - df[close_ask]
    df[cldiff_ask] = df[close_ask] - df[low_ask]
    df[hldiff_ask] = df[high_ask] - df[low_ask]

    df[hpcdiff_ask] = df[high_ask] - df[close_ask].shift(1)
    df[cpldiff_ask] = df[close_ask].shift(1) - df[low_ask]
    df[hpldiff_ask] = df[high_ask].shift(1) - df[low_ask]

    high_bid = instrument + '_highBid'
    low_bid = instrument + '_lowBid'
    close_bid = instrument + '_closeBid'
    hcdiff_bid = 'Diff High-Close_bid'
    cldiff_bid = 'Diff Close-Low_bid'
    hldiff_bid = 'Diff High-Low_bid'
    hpcdiff_bid = 'Diff High-PastClose_bid'
    cpldiff_bid = 'Diff PastClose-Low_bid'
    hpldiff_bid = 'Diff PastHigh-Low_bid'

    df[hcdiff_bid] = df[high_bid] - df[close_bid]
    df[cldiff_bid] = df[close_bid] - df[low_bid]
    df[hldiff_bid] = df[high_bid] - df[low_bid]

    df[hpcdiff_bid] = df[high_bid] - df[close_bid].shift(1)
    df[cpldiff_bid] = df[close_bid].shift(1) - df[low_bid]
    df[hpldiff_bid] = df[high_bid].shift(1) - df[low_bid]

    df = df[1:]
    df[date] = df[date].astype(str)
    df[date] = df[date].str[:13]
    
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df

def get_indicators(data, instrument, column, wind, bidask):
    """
    Rolling Mean, Bollinger Bands and RSI Calculations

    """
    # Finance Inidcators
    df = data.copy()
    df['rolling_mean_{}_{}'.format(column, wind)] = df['{}_close{}'.format(instrument, bidask)].rolling(window=wind).mean()
    df['rolling_std_{}_{}'.format(column, wind)] = df['{}_close{}'.format(instrument, bidask)].rolling(window=wind).std()
    df['rolling_sum_{}_{}'.format(column, wind)] = df['{}_close{}'.format(instrument, bidask)].rolling(window=wind).sum()
    df['bollinger_{}_up_{}'.format(column, wind)] = df['rolling_mean_{}_{}'.format(column, wind)] + 2*df['rolling_std_{}_{}'.format(column, wind)]
    df['bollinger_{}_low_{}'.format(column, wind)] = df['rolling_mean_{}_{}'.format(column, wind)] - 2*df['rolling_std_{}_{}'.format(column, wind)]
    df['bollinger_{}_up_{}_surpassed'.format(column, wind)] = 0
    df.loc[df['bollinger_{}_up_{}'.format(column, wind)] < df[column], 'bollinger_{}_up_{}_surpassed'.format(column, wind)] = 1
    df['bollinger_{}_low_{}_surpassed'.format(column, wind)] = 0
    df.loc[df['bollinger_{}_low_{}'.format(column, wind)] > df[column], 'bollinger_{}_low_{}_surpassed'.format(column, wind)] = 1
    df['rolling_mean_{}_{}_surpassed'.format(column, wind)] = 0
    df.loc[df['rolling_mean_{}_{}'.format(column, wind)] < df[column], 'rolling_mean_{}_{}_surpassed'.format(column, wind)] = 1

    # Other
    df['Close/Open'] = df[column]/df[column]

    for i in [c for c in df.columns if bidask in c]:

        df[i + 'lag1'] = df[i].shift(1)
        df[i + 'lag2'] = df[i].shift(2)
        df[i + 'lag3'] = df[i].shift(3)
        df[i + 'lag4'] = df[i].shift(4)
        df[i + 'lag5'] = df[i].shift(5)


    return df
