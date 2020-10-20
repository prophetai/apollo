import os
import logging
import pandas as pd
import numpy as np
import oandapy as opy
from datetime import datetime as dt
from sqlalchemy import create_engine
from loadAssets import Assets
import json



def save_order(account,model,instrument,order,probability,conn_data):
    """
    Saves the desition made from the predictions

    Args:
        - account(str): nickname of destination account
        - model(str): model used
        - instrument(str): instrument traded
        - order(Order): order data from broker
        - probability(float): probability from decision
        - conn_data(dic): dictionary to connect to the database
            conn_data = {
            'db_user': ,
            'db_pwd':  ,
            'db_host': ,
            'db_name': 
    }
    """
    if order.trade.units > 0: # We are making a Buy
        trade = "buy"
    elif order.trade.units < 0: # We are making a Sell
        trade = "sell"   
    else: # We are not making a trade
        logging.info('Neutral - No decision to save')
        return

    account_type = "live"
    trading_url = os.environ['trading_url_'+ account]
    account = os.environ['trading_url_'+ account].split('/')[-2]
    if 'practice' in trading_url:
        account_type = "practice"
    
    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')
    
    data = pd.DataFrame(data = {"account": account,
                            "order_id": [order.i_d],
                            "account_type": [account_type],
                            "entry_price":[order.entry_price],
                            "ask": [order.ask_price],
                            "bid": [order.bid_price],
                            "instrument": [order.trade.instrument],
                            "model": [model],
                            "units": [order.trade.units],
                            "probability": [probability],
                            "stop_loss": [order.trade.stop_loss],
                            "take_profit": [float(order.trade.take_profit)],
                            "time": [dt.now()],
                            "trade": [trade]})
    logging.info('Data to save on Database')
    logging.info(data.reset_index(drop=True).to_dict())
    data = data.reset_index(drop=True)
    try:
        data.to_sql('trades', engine, if_exists="append",index=False)
    except Exception as e:
        logging.error(e)
    
def save_input(conn_data, account, model_version, current_time, inv_instrument, original_datasets, order_id=None):
    """
    Saves model input data to Database

    -
    """
    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
        
    assets = Assets(model_version, inv_instrument)
    variablesh, variablesl = assets.load_vals()

    data_high = {variablesh[i]: original_datasets[0][0][i] for i in range(len(original_datasets[0][0]))}
    data_low = {variablesl[i]: original_datasets[1][0][i] for i in range(len(original_datasets[1][0]))}
    
    df_todb = pd.DataFrame()
    df_todb['account'] = [os.environ['trading_url_'+ account].split('/')[-2]]
    df_todb['model_version'] = [model_version]
    df_todb['date'] = [current_time]
    df_todb['order_id'] = [order_id]
    df_todb['data_high'] = [json.dumps(data_high)]
    df_todb['data_low'] = [json.dumps(data_low)]
    
    
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')
    try:
        df_todb.to_sql(f'historical_datasets',
            engine, if_exists="append", index=False)
    except Exception as e:
        logging.error(e)

    