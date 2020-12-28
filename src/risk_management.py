from datetime import datetime as dt
import os
import requests
import logging
from trade import Trade
from trade.order import Order


def check_stop_loss(trades_list, account):
    """
    Checks if active trades need stop-losses
    """
    for element in trades_list:
        trade = Trade(element['instrument'],
                        element['initialUnits'],
                        i_d=element['id'],
                        price=element['price'],
                        take_profit=element['takeProfitOrder']['price'],
                        account=os.environ['trading_url_' + account].split('/')[-2],
                        openTime=element['openTime'])
        duration = trade.get_trade_duration()

        if duration >= 3 and 'stopLossOrder' not in element:
            logging.info(f'Trade to set SL:{trade.i_d} [{duration}h]')
            try:
                trade.get_stop_loss()
                stop_loss = trade.stop_loss
                order = Order(trade, account)
                order.set_stop_loss(stop_loss)
                logging.info(f'SL [{stop_loss}] set for trade ({trade.i_d})')
            except Exception as e:
                logging.error(f'No se pudo obtener trade [{trade.i_d}] de la BD ({e})')
            