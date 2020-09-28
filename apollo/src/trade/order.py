import requests
import os
import logging
from trade import Trade

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    #filename='log.txt'
)

class Order():

    def __init__(self,trade,account):
        """Clase que crea orden en Broker.

        Args:
            inv_instrument (str): instrument's name
            take_profit (float): take profit price
        """
        self.trade = trade
        self.auth_token = os.environ['token'] #token de autenticación
        self.head = {'Authorization': 'Bearer ' + self.auth_token} # header
        self.account = account
        self.url = os.environ['trading_url_'+account] # URL de broker
        self.i_d = None
        self.entry_price = None
        self.ask_price = None
        self.bid_price = None
        self.spread = None

    def make_market_order(self):
        data = {
                "order" :{
                    "type": "MARKET",
                    "instrument": self.trade.instrument,
                    "units": str(self.trade.units),
                    "takeProfitOnFill":{
                        "price": str(self.trade.take_profit)
                    }
                }
            }
            
        url = self.url + 'orders'
        try:
            response = requests.post(url, json=data, headers=self.head)
            print(f'Response code for Order sent:\n{response}')
            self.broker_data = response.json()
            print(f'Content of response:\n{self.broker_data}')
            self.i_d = self.broker_data['orderFillTransaction']['tradeOpened']['tradeID']
            self.entry_price = self.broker_data['orderFillTransaction']['tradeOpened']['price']
            self.ask_price = self.broker_data['orderFillTransaction']['fullPrice']['closeoutAsk']
            self.bid_price = self.broker_data['orderFillTransaction']['fullPrice']['closeoutBid']
            self.spread = float(self.broker_data['orderFillTransaction']['tradeOpened']['halfSpreadCost']) * 2
            # TO-DO #81 self.trade.time = self.broker_data['orderFillTransaction']['time']
        except Exception as e:
            logging.error(f'Error in make_market_order')
            logging.error(e)
        

    def set_stop_loss(self):
        data = {
            "stopLoss": {
                "timeInForce": "GTC",
                "price": f"{self.trade.stop_loss}"
                }}

        url = f'{self.url}trades/{self.trade.i_d}/orders'
        response = requests.put(url, json=data, headers=self.head)
        json_response = response.json()
        print(f'Content of response:\n{json_response}')

if __name__ == "__main__":
    new_trade = Trade('USD_JPY',1000,take_profit=110.90)
    new_order = Order(new_trade,'1h')
    new_order.make_market_order()   