import requests
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    #filename='log.txt'
)

class Order():

    def __init__(self, inv_instrument, take_profit, stop_loss):
        """Inicializa nuestra red.

        Args:
            nn_param_candidatos (dict): parámetros que puede incluir la red.
            Por ejemplo:
                num_neurons (list): [64, 128, 256]
                num_capas (list): [1, 2, 3, 4]
                activacion (list): ['relu', 'elu']
                optimizador (list): ['rmsprop', 'adam']
        """

        self.auth_token = os.environ['token'] #token de autenticación
        self.hed = {'Authorization': 'Bearer ' + self.auth_token} # header
        self.url = os.environ['trading_url'] # URL de broker
        self.tradeID = 0
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.market_price = 0
        self.inv_instrument = inv_instrument

    def make_market_order(self, units):
        data = {
            "order" :{
            	"type": "MARKET",
            	"instrument": self.inv_instrument,
            	"units": units,
                "takeProfitOnFill":{
                    "price": self.take_profit
                },
                "stopLossOnFill":{
                    "price": self.stop_loss
                }
            }
        }


        url = os.environ['trading_url']
        #try:
        response = requests.post(url, json=data, headers=self.hed)
        print(response)
        json_response = response.json()
        print(json_response)
        self.tradeID = json_response['orderCreateTransaction']['id']
        #self.market_price = json_response['orderCreateTransaction']['price']
        #except requests.exceptions.RequestException as e:
            #logging.info(e)
            #sys.exit(1)

        return json_response


    def show_order(self):
        print(f"""\ntradeID: {self.tradeID}\
                  \nmarket_price: {self.market_price}\
                  \nstop_loss: {self.stop_loss}\
                  \ntake_profit: {self.take_profit}""")
