import pandas as pd
import datetime
import trade


def check_stop_loss(trades_list):
    for data in trades_list:
       current_trade = trade(data)
       if current_trade.get_trade_time() >= 3 and not current_trade.get_stop_loss:
           current_trade.get_stop_loss()
           current_trade.set_stop_loss()
        