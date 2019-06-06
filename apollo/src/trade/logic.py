# coding: utf-8
import numpy as np
import pandas as pd
from .pip_calculator import get_profit, get_loss

class Decide:
    
    def __init__(self, data_buy, data_sell, portfolio, direction=0, magnitude=0, take_profit=0 , stop_loss=0):
        """
        Makes a decision about the next operation

        Args:
            - data (Dataframe): gett

        """
        self.data_buy_tp = data_buy
        self.data_sell_tp = data_sell
        self.data_buy_sl = pd.DataFrame()
        self.data_sell_sl = pd.DataFrame()
        self.direction = direction
        self.magnitude = magnitude
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.portfolio = portfolio
        self.decision = ''
        self.spread = 0.15
    
    def get_all_pips(self):
        data_buy_tp = self.data_buy_tp
        data_sell_tp = self.data_sell_tp
        data_buy_sl = self.data_buy_sl
        data_sell_sl = self.data_sell_sl
        portfolio = self.portfolio

        data_buy_tp['portfolio_gain'] = get_profit(data_buy_tp['Open'], data_buy_tp['Take Profit'], 1) + portfolio
        data_buy_sl['portfolio_loss'] = get_loss(data_sell_tp['Open'], data_sell_tp['Take Profit'], 1) + portfolio
        data_buy_sl['Stop Loss'] = data_sell_tp['Take Profit']

        data_sell_tp['portfolio_gain'] = get_profit(data_sell_tp['Open'], data_sell_tp['Take Profit'], 1) + portfolio
        data_sell_sl['portfolio_loss'] = portfolio - get_loss(data_buy_tp['Open'], data_buy_tp['Take Profit'], 1) 
        data_sell_sl['Stop Loss'] = data_buy_tp['Take Profit']
        

        data_buy_tp['utility_gain'] = data_buy_tp['Probability'] * np.log(data_buy_tp['portfolio_gain'])
        data_sell_tp['utility_gain'] = data_sell_tp['Probability'] * np.log(data_sell_tp['portfolio_gain'])
        
        buy_losses = portfolio - get_profit(data_sell_tp['Open'], data_sell_tp['Take Profit'], 1)
        data_buy_sl['utility_loss'] = data_sell_tp['Probability'] * np.log(buy_losses)
        data_buy_sl['Probability'] = data_sell_tp['Probability']
        
        sell_losses = portfolio - get_profit(data_buy_tp['Open'], data_buy_tp['Take Profit'], 1)
        data_sell_sl['utility_loss'] = data_buy_tp['Probability'] * np.log(sell_losses)
        data_sell_sl['Probability'] = data_buy_tp['Probability']

        buy_decision_tp = data_buy_tp.loc[data_buy_tp['utility_gain'].idxmax()]
        buy_decision_sl = data_buy_sl.loc[data_buy_sl['utility_loss'].idxmax()]

        print(buy_decision_tp, buy_decision_sl)
        decision_buy = buy_decision_tp['utility_gain'] + buy_decision_sl['utility_loss']

        sell_decision_tp = data_sell_tp.loc[data_sell_tp['utility_gain'].idxmax()]
        sell_decision_sl = data_sell_sl.loc[data_sell_sl['utility_loss'].idxmax()]
        decision_sell = sell_decision_tp['utility_gain'] + sell_decision_sl['utility_loss']

        if decision_buy > decision_sell: #and buy_decision_tp['portfolio_gain'] - portfolio >  self.spread :
            self.decision = self.decision + '\n Buy!\n' + str(buy_decision_tp) + str(buy_decision_sl) + f'\n Expected Utility: {decision_buy}'
            self.direction = 1
            self.take_profit = buy_decision_tp['Take Profit']
            self.stop_loss = buy_decision_sl['Stop Loss']
        elif decision_buy < decision_sell: #and sell_decision_tp['portfolio_gain'] - portfolio  >  self.spread :
            self.decision = self.decision +'\n Sell!\n' + str(sell_decision_tp) + str(sell_decision_sl) + f'\n Expected Utility: {decision_sell}'
            self.direction = -1
            self.take_profit = sell_decision_tp['Take Profit']
            self.stop_loss = sell_decision_sl['Stop Loss']
        #else:
            #self.decision = 'Neutral'

if __name__ == '__main__':
    data_buy = {'Open': [108.09, 108.09, 108.09, 108.09, 108.09],
                'Take Profit': [108.097, 108.107, 108.118, 108.129, 108.140],
                'Probability': [0.896, 0.796, 0.671, 0.594, 0.339],
                'bucket': [58.67, 59.59, 59.26, 43.75, 23.98]}
    data_sell = {'Open': [108.09, 108.09, 108.09, 108.09, 108.09],
                'Take Profit': [108.080, 108.069, 108.058, 108.047, 108.037],
                'Probability': [0.759, 0.677, 0.442, 0.362, 0.322],
                'bucket': [58.67, 59.59, 59.26, 43.75, 23.98]}
    df_buy = pd.DataFrame(data_buy)
    df_sell = pd.DataFrame(data_sell)  
    portfolio = 100
    decision = Decide(df_buy, df_sell, portfolio, direction=0, magnitude=0, take_profit=0 , stop_loss=0)
    decision.get_all_pips()
    print(f'Buy Take Profit:\n{decision.data_buy_tp}\n\nBuy Stop Loss:\n{decision.data_buy_sl}')
    print(f'Sell Take Profit:\n{decision.data_sell_tp}\n\nSell Stop Loss:\n{decision.data_sell_sl}')
    print(f'\nDecision: {decision.decision}')