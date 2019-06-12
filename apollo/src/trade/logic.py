# coding: utf-8
import numpy as np
import pandas as pd
from .pip_calculator import get_profit, get_loss
import time

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
        self.spread = 0.075
    
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


        # Se toman los TP y SL para Buy y Sell que tengan la mayor utilidad y cubran el spread

        data_buy_tp_aux = data_buy_tp.copy()
        data_buy_sl_aux = data_buy_sl.copy()
        greater_spread = False

        # Buy TP
        buy_decision_tp_idx = 0
        buy_decision_tp = 0
        buy_gain = 0

        while not greater_spread: # mientras que la ganancia no cubra el spread next            
            buy_decision_tp_idx = data_buy_tp_aux['utility_gain'].idxmax()
            buy_decision_tp = data_buy_tp_aux.loc[buy_decision_tp_idx]
            buy_gain = buy_decision_tp['portfolio_gain'] - self.portfolio
            print(f'buy_gain: {buy_gain}')

            if buy_gain > self.spread:
                greater_spread = True
            else:
                print(f'To delete:{buy_decision_tp_idx}')
                data_buy_tp_aux.drop(buy_decision_tp_idx, inplace=True)
        
        # Buy SL
        greater_spread = False
        buy_decision_sl_idx = 0
        buy_decision_sl = 0
        buy_loss = 0
        
        while not greater_spread: # mientras que la perdida no cubra el spread next
            buy_decision_sl_idx = data_buy_sl_aux['utility_loss'].idxmax()
            buy_decision_sl = data_buy_sl_aux.loc[buy_decision_sl_idx]
            buy_loss = abs(self.portfolio - buy_decision_sl['portfolio_loss'])
            print(f'buy_loss: {self.portfolio} - {buy_loss}')
            if buy_loss > self.spread:
                greater_spread = True
            else:
                data_buy_sl_aux.drop(buy_decision_sl_idx, inplace=True)

        # Sell TP
        data_sell_tp_aux = data_sell_tp.copy()
        data_sell_sl_aux = data_sell_sl.copy()
        greater_spread = False

        sell_decision_tp_idx = data_sell_tp_aux['utility_gain'].idxmax()
        sell_decision_tp = data_sell_tp_aux.loc[sell_decision_tp_idx]
        sell_gain = self.portfolio - sell_decision_tp['portfolio_gain']
        
        while not greater_spread:
            sell_decision_tp_idx = data_sell_tp_aux['utility_gain'].idxmax()
            sell_decision_tp = data_sell_tp_aux.loc[sell_decision_tp_idx]
            sell_gain = sell_decision_tp['portfolio_gain'] - self.portfolio
            print(f'sell_gain: {sell_gain}')
            if sell_gain > self.spread:
                greater_spread = True
            else:
                data_sell_tp_aux.drop(sell_decision_tp_idx, inplace=True)
        # Sell SL
        greater_spread = False

        sell_decision_sl_idx = 0
        sell_decision_sl = 0
        sell_loss = 0

        while not greater_spread:
            sell_decision_sl_idx = data_sell_sl_aux['utility_loss'].idxmax()
            sell_decision_sl = data_sell_sl_aux.loc[sell_decision_sl_idx]
            sell_loss = abs(self.portfolio - sell_decision_sl['portfolio_loss'])
            print(f'sell_loss: {sell_loss}')
            if sell_loss > self.spread:
                greater_spread = True
            else:
                data_sell_sl_aux.drop(sell_decision_sl_idx, inplace=True)

        decision_buy = buy_decision_tp['utility_gain'] + buy_decision_sl['utility_loss']
        print(f'\nBest course of action for Buy: TP: \n{buy_decision_tp}, SL: {buy_decision_sl}')
        print(f'Buy gain: {buy_gain}, Buy Loss: {buy_loss}')
        print(f'Expected utility: {decision_buy}')
        
        decision_sell = sell_decision_tp['utility_gain'] + sell_decision_sl['utility_loss']
        print(f'\nBest course of action for Sell: TP: \n{sell_decision_tp}, SL: {sell_decision_sl}')
        print(f'Sell gain: {sell_gain}, Sell Loss:{sell_loss}')
        print(f'Expected utility: {decision_sell}')

        if decision_buy > decision_sell and buy_decision_tp['portfolio_gain'] - portfolio >  self.spread :
            self.decision = self.decision + '\n Buy!\n' + str(buy_decision_tp) + str(buy_decision_sl) + f'\n Expected Utility: {decision_buy}'
            self.direction = 1
            self.take_profit = round(buy_decision_tp['Take Profit'] + self.spread/2,3) # IMPORTANT!! SPREAD WAS ADDED
            self.stop_loss = round(buy_decision_sl['Stop Loss'] - self.spread/2,3)
        elif decision_sell > decision_buy and sell_decision_tp['portfolio_gain'] - portfolio  >  self.spread :
            self.decision = self.decision +'\n Sell!\n' + str(sell_decision_tp) + str(sell_decision_sl) + f'\n Expected Utility: {decision_sell}'
            self.direction = -1
            self.take_profit = round(sell_decision_tp['Take Profit'] - self.spread/2, 3)
            self.stop_loss = round(sell_decision_sl['Stop Loss'] + self.spread/2, 3)   
        else:
            self.decision = f'Neutral \nBuy utility:{decision_buy} \nSell utility: {decision_sell})'
            self.decision += f"\nBuy Gain ${sell_decision_tp['portfolio_gain'] - portfolio}"
            self.decision += f"\nSell Gain ${sell_decision_tp['portfolio_gain'] - portfolio}"


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