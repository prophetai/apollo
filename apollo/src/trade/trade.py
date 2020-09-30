import requests
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

class Trade():

    def __init__(self, instrument, units, i_d=None, price=None, take_profit=None, openTime=None):
        """Inicializa un trade
        Args:
            instrument (str): investment instrument name
            max_units (float): maximum quantity of units that can be traded at the same time
        """
        self.i_d = i_d
        self.instrument = instrument
        self.units = units
        self.price = price
        self.take_profit = take_profit
        self.stop_loss = None
        self.openTime = openTime
        
    
    def get_stop_loss(self):
        self.stop_loss = round(2.5 * float(self.price) - 1.5 * float(self.take_profit),3)

    

if __name__ == "__main__":
    trade = Trade(1,'USD_JPY',110.101, 2000, 111.00)
    trade.get_stop_loss()

    logging.info(f'ID:{trade.i_d} \nUnits:{trade.units} \nStop Loss:{trade.stop_loss}')