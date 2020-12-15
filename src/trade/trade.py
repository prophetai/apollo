import os
import logging
import requests
import math
from sqlalchemy import orm, Column, String, Integer, Numeric, DateTime
from saveToDB import get_trade_from_id
from datetime import datetime as dt
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)


class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)

    def __init__(self, instrument, units, i_d=None, price=None, account=None, take_profit=None, openTime=None):
        """Inicializa un trade
        Args:
            instrument (str): investment instrument name
            max_units (float): maximum quantity of units that can be traded at the same time
        """
        self.__table_name__ = 'trades'
        self.i_d = i_d
        self.instrument = instrument
        self.units = units
        self.price = price
        self.take_profit = take_profit
        self.stop_loss = None
        self.openTime = openTime
        self.account = account

    def __repr__(self):

        return f"<Trade(id={self.i_d},\
                instrument = {self.instrument},\
                stop_loss = {self.stop_loss},\
                take_profit = {self.take_profit},\
                units = {self.units}>"

    def get_stop_loss(self):
        """
        Gets stop loss for current trade
        take_profit_pips = abs(self.take_profit - self.price)
        sl_break_even = (((self.units * (take_profit_pips * 0.01/self.take_profit)) * probability * self.price)
        ----------------------------------------------------------------------------------------
                          ((1-probability)*self.units *0.01)      
        """
        probability = self.__get_probability()
        take_profit = float(self.take_profit)
        price = float(self.price)
        units = float(self.units)
        take_profit_pips = abs(take_profit - price)

        inv_direction = price - take_profit # gets the sign of the SL (inverted)
        calc_sl = (((units * (take_profit_pips * 0.01/take_profit)) * probability * price) / ((1-probability) * units * 0.01))
        
        self.stop_loss = round(price + math.copysign(calc_sl, inv_direction),3)


    def get_trade_duration(self):
        """
        Gets how long the trade has been open in hours.
        """
        date_time_obj = dt.strptime(self.openTime[:-4], '%Y-%m-%dT%H:%M:%S.%f')
        duration = dt.now() - date_time_obj
        duration_in_s = abs(duration.total_seconds())
        hours = abs(divmod(duration_in_s, 3600)[0])

        return hours

    def __get_probability(self):
        """
        Gets the probability of success from the trade's DB
        """
        data = get_trade_from_id(self.i_d, self.account)
        probability = round(float(data.probability), 2)

        return probability


if __name__ == "__main__":
    trade = Trade(1, 'USD_JPY', 110.101, 2000, 111.00)
    probability = 0.7
    trade.get_stop_loss(probability)

    logging.info(
        f'ID:{trade.i_d} \nUnits:{trade.units} \nStop Loss:{trade.stop_loss}')
