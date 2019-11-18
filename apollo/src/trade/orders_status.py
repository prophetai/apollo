import requests
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

class Positions():

    def __init__(self, inv_instrument):
        """Inicializa nuestra red.

        Args:
            inv_instrument (str): instrument name
            max_units (float): maximum quantity of units that can be traded at the same time
        """
        self.inv_instrument = inv_instrument
        self.long_units = 0
        self.short_units = 0

    def get_status(self):
        auth_token = os.environ['token'] #token de autenticación
        head = {'Authorization': 'Bearer ' + auth_token} # header
        url = os.environ['trading_url'] + 'positions'
        
        response = requests.get(url, headers=head)
        
        print(f'Response code for Order sent:\n{response}')
        json_response = response.json()
        
        for positions in json_response['positions']:            
            if positions['instrument'] == self.inv_instrument:
                self.long_units += float(positions['long']['units'])
                self.short_units += abs(float(positions['short']['units']))

if __name__ == "__main__":
    positions = positions('USD_JPY')
    positions.get_status()

    print(f'Long:{positions.long_units} units, Short:{positions.short_units} units')