from datetime import datetime as dt
import os
import requests
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
                        openTime=element['openTime'])
        
        if trade.get_trade_duration() >= 3 and 'stopLossOrder' not in element:
            print(f'Duration:{trade.get_trade_duration()}')
            print(f'Trade found for SL:{trade.i_d}')
            trade.get_stop_loss()
            stop_loss = trade.stop_loss
            order = Order(trade, account)
            order.set_stop_loss(stop_loss)