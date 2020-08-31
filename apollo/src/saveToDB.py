import os
import logging
import pandas as pd
import numpy as np
import oandapy as opy
from datetime import datetime as dt
from sqlalchemy import create_engine




def save_decisions(account, model, instrument, decision, conn_data):
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
    account_type = "live"
        if 'practice' in os.environ['trading_url_'+ account]:
            account_type = "practice"
    s = decision.decision
    ask = decision.data_buy["Open"][0]
    bid = decision.data_sell["Open"][0]
    user = conn_data['db_user']
    pwd = conn_data['db_pwd']
    host = conn_data['db_host']
    data_base = conn_data['db_name']
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:5432/{data_base}')

    if decision.direction == 1: # We are making a Buy 
        probability = float(s.partition("Sell")[0].partition("Probability")[2][2:7])/100
        prediction_used = decision.data_buy[(decision.data_buy["Probability"] >= probability - 0.001) & (decision.data_buy["Probability"] < probability + 0.001)].index[0]
        stop_loss = np.nan
        take_profit = float(decision.take_profit)
        time = dt.now()        
        trade = "buy"

    #elif decision.direction == -1: # We are making a Sell
    else:
        probability = float(s.partition("Sell")[2].partition("Probability")[2][2:7])/100
        prediction_used = decision.data_sell[(decision.data_sell["Probability"] >= probability - 0.001) & (decision.data_sell["Probability"] < probability + 0.001)].index[0]
        stop_loss = np.nan
        take_profit = float(decision.take_profit)
        time = dt.now()
        trade = "sell"
    
    data = pd.DataFrame({"account": account,
                            "account_type": account_type,
                            "ask": ask,
                            "bid": bid,
                            "instrument": instrument,
                            "model": model,
                            "prediction_used": prediction_used,
                            "probability": probability,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "time": time,
                            "trade": trade}, index=[dt.now()])
    
    data = data.reset_index(drop=True)
    data.to_sql('trades', engine, if_exists="append")
