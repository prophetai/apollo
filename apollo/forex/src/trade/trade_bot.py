#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import alpaca_trade_api as tradeapi


def connect_to_alpaca(api_key,secret_key, debug=False):
    """
    Hace la conexión a la cuenta de alpaca

    Args:
        - api_key(string): identificador de la cuenta
        - secret_key(string): pass de la cuenta
    Returns:
        - api(alpaca_trade_api.rest.REST): objeto de la conexión a alpaca
    """
    if debug:
        api = tradeapi.REST(api_key,secret_key, 'https://paper-api.alpaca.markets')
    else:
        api = tradeapi.REST(api_key,secret_key)
    account = api.get_account()
    api.list_positions()
    print(f'Type: {type(api)}\nConnected:{account}')

    return api

if __name__ == '__main__':
    api_key = os.environ['ALPACA_API_KEY']
    api_secret = os.environ['ALPACA_SECRET_KEY']

    # Connect
    api=connect_to_alpaca(api_key, api_secret, debug=True)
    # Submit Order
    symbol = input("Symbol?")
    qty = input("Quantity?")
    side = input("Side?")
    type = input("type?")
    time_in_force = input("time in force?")

    '''
    day
    The order is good for the day, and it will be canceled automatically at the end of market hours.
    gtc
    The order is good until canceled.
    opg
    The order is placed at the time the market opens.
    '''

    order = api.submit_order(symbol, qty, side, type, time_in_force, limit_price=None, stop_price=None, client_order_id=None)
    print(order)
