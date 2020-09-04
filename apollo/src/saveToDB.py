import os
import logging
import pandas as pd
import numpy as np
import oandapy as opy
from datetime import datetime as dt
from sqlalchemy import create_engine




def save_decisions(account, model, instrument, decision, conn_data, units):
    """
    Saves the desition made from the predictions

    Args:
        - account(str): nickname of destination account
        - model(str): model used
        - instrument(str): instrument traded
        - decision(str): decision made
        - conn_data(dic): dictionary to connect to the database
            conn_data = {
            'db_user': ,
            'db_pwd':  ,
            'db_host': ,
            'db_name': 
    }
    """
    if decision.direction == 1: # We are making a Buy
        trade = "buy"
    elif decision.direction == -1: # We are making a Sell
        trade = "sell"   
    else: # We are not making a trade
        logging.info('Neutral - No decision to save')
        return

    account_type = "live"
    if 'practice' in os.environ['trading_url_'+ account]:
        account_type = "practice"
    
    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')
    probability = decision.probability
    account = str(os.environ['trading_url_'+ account]).split('/')[-2]
    
    data = pd.DataFrame({"account": account,
                            "account_type": account_type,
                            "ask": decision.data_buy["Open"][0],
                            "bid": decision.data_sell["Open"][0],
                            "instrument": instrument,
                            "model": model,
                            "units": units,
                            "probability": probability,
                            "stop_loss": decision.stop_loss,
                            "take_profit": float(decision.take_profit),
                            "time": dt.now(),
                            "trade": trade}, index=[dt.now()])
    logging.info('Data to save on Database')
    logging.info(data.reset_index(drop=True).to_dict())
    data = data.reset_index(drop=True)
    try:
        data.to_sql('trades', engine, if_exists="append",index=False)
    except Exception as e:
        logging.error(e)
