import requests
import os
import logging
from trade import Trade

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

class openTrades():

    def __init__(self,account):
        """Inicializa los trades abiertos

        Args:
            instrument (str): investment instrument name
            max_units (float): maximum quantity of units that can be traded at the same time
        """
        self.trades = []
        self.head = {'Authorization': 'Bearer ' + os.environ['token']} # header
        self.url = f"{ os.environ['trading_url_'+ account]}openTrades/"
        self.account = account

    def get_trades_data(self):
        """
        Gets the id and units of an open trade
        """
        
        response = requests.get(self.url, headers=self.head)
        json_response = response.json()
        for trade in json_response['trades']:
            units = float(trade['currentUnits'])
            i_d = trade['id']
            take_profit = trade['takeProfitOrder']['price']
            instrument = trade['instrument']
            price = trade['price']
            openTime = trade['openTime']
            new_trade = Trade(instrument, units, i_d=i_d, price=price, take_profit=take_profit, openTime=openTime)
            self.trades.append(new_trade)
    
    def number_trades(self):
        auth_token = os.environ['token'] #token de autenticación
        head = {'Authorization': 'Bearer ' + auth_token} # header
        url = os.environ['trading_url_' + self.account] + 'positions'
        
        response = requests.get(url, headers=head)
        positions = response.json()
        
        number = 0
        for position in positions['positions']:
            number += float(position['long']['units'])
            number += abs(float(position['short']['units']))

        return number


if __name__ == "__main__":
    open_trades = openTrades('1h')
    open_trades.get_trades_data()
    print(f'Number of trades open: {open_trades.number_trades()}')

    for trade in open_trades.trades:
        trade.get_stop_loss()
        print(f'ID:{trade.i_d} \n\
            Instrument: {trade.instrument}\n\
            Price: {trade.price}\n\
            Units:{trade.units} \n\
            Take Profit:{trade.take_profit}\n\
            Stop Loss:{trade.stop_loss}')