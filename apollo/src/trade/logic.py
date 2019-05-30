# coding: utf-8
import numpy as np
import pandas as pd
from pip_calculator import get_profit

class Decide:
    
    def __init__(self, data_buy, data_sell, portfolio, direction=0, magnitude=0, take_profit=0 , stop_loss=0):
        """
        Makes a decision about the next operation

        Args:
            - data (Dataframe): gett

        """
        self.data_buy = data_buy
        self.data_sell = data_sell
        self.direction = direction
        self.magnitude = magnitude
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.portfolio = portfolio
    
    def get_all_pips(self):
        data_buy = self.data_buy
        data_sell = self.data_sell
        portfolio = self.portfolio

        data_buy['portfolio_gain'] = get_profit(data_buy['Open'], data_buy['Take Profit'], 1) + portfolio
        data_sell['portfolio_gain'] = get_profit(data_sell['Open'], data_sell['Take Profit'], 1) + portfolio

        # esto está mal, hay que utilizar el del otro dataframe
        data_buy['portfolio_loss'] = portfolio - get_profit(data_buy['Open'], data_buy['Take Profit'], 1) 
        data_sell['portfolio_loss'] = portfolio - get_profit(data_sell['Open'], data_sell['Take Profit'], 1) 

        data_buy['utility_gain'] = data_buy['Probability'] * np.log(data_buy['portfolio_gain'])
        data_buy['utility_loss'] = data_buy['Probability'] * np.log(data_buy['portfolio_loss'])
        
        data_sell['utility_gain'] = data_sell['Probability'] * np.log(data_sell['portfolio_gain'])
        data_sell['utility_loss'] = data_sell['Probability'] * np.log(data_sell['portfolio_loss'])
        


        self.data_buy = data_buy
        self.data_sell = data_sell



if __name__ == '__main__':
    data_buy = {'Open': [109.5, 109.5, 109.5, 109.5],
                'Take Profit': [109.575, 109.586, 109.597, 109.608],
                'Probability': [1.0, 0.789, 0.701, 0.584],
                'bucket': [58.67, 59.59, 59.26, 43.75]}
    data_sell = {'Open': [109.5, 109.5, 109.5, 109.5],
                'Take Profit': [109.575, 109.586, 109.597, 109.608],
                'Probability': [1.0, 0.789, 0.701, 0.584],
                'bucket': [58.67, 59.59, 59.26, 43.75]}
    df_buy = pd.DataFrame(data_buy)
    df_sell = pd.DataFrame(data_sell)  
    portfolio = 100
    decision = Decide(df_buy, df_sell, portfolio, direction=0, magnitude=0, take_profit=0 , stop_loss=0)
    decision.get_all_pips()
    
    print(f'Buy:\n{decision.data_buy}\n\nSell:\n{decision.data_sell}')