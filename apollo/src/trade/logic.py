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
        self.data_buy = data_buy
        self.data_sell = data_sell
        self.direction = direction
        self.magnitude = magnitude
        self.take_profit = take_profit
        self.decision = ''
        self.spread = 0.15
    
    def get_best_action(self, buy_sell):
        if buy_sell = 'Buy':
            data = self.data_buy
        elif buy_sell = 'Sell':
            data = self.data_sell
        greater_spread = False

        last_idx = len(data) - 1
        probability = 0
        profit = 0
        spread = self.spread

        print(f'\nSearching for best {buy_sell} strategy(TP):')
        # Mientras la proba no sea >0.5, no se nos acabe la tabla y la ganancia sea mayor que el spread sigue buscando 
        while probability <= 0.5 or last_idx > -1 or profit-spread < 0:
            best_action = data.loc[last_idx]
            probability = best_action['Probability']
            profit = abs(best_action['Open'] - best_action['Take Profit'])
            last_idx -= 1
        
        if last_idx > -1: # si logra encontrar un ganador
            return best_action, profit
        else:
            return data.loc[0], 0 


    
    def get_all_pips(self):
        data_buy_tp = self.data_buy_tp
        data_sell_tp = self.data_sell_tp
        data_buy_sl = self.data_buy_sl
        data_sell_sl = self.data_sell_sl
        portfolio = self.portfolio


        # Se toman los TP y SL para Buy y Sell que tengan la mayor utilidad y cubran el spread
        data_buy_tp_aux = data_buy_tp.copy()
        data_buy_sl_aux = data_sell_tp.copy()
        greater_spread = False

        # Se calcula decisión
        best_action_buy, profit_buy = get_best_action('Buy')
        best_action_sell, profit_sell = get_best_action('Sell')

        if profit_buy > profit_sell:
            self.decision = self.decision + '\n Buy!\n' + str(buy_decision_tp) + str(buy_decision_sl) + f'\n Expected Utility: {decision_buy}'
            self.direction = 1
            self.take_profit = round(profit_buy, 3)
        elif profit_sell > profit_buy:
            self.decision = self.decision +'\n Sell!\n' + str(sell_decision_tp) + str(sell_decision_sl) + f'\n Expected Utility: {decision_sell}'
            self.direction = -1
            self.take_profit = round(profit_sell, 3) # adjusted for spread
        else:
            self.decision = '\nNeutral \nBuy '
            self.decision += f"\nBuy Gain: ${round(profit_buy,3)}"
            self.decision += f"\nProbability TP:{best_action_buy['Probability']}"
            self.decision += f"\nSell Gain ${round(profit_buy,3)}"
            self.decision += f"\nProbability TP:{sell_decision_tp['Probability']}"
        
        #data_buy_tp.drop(['Portfolio Gain'], axis=1, inplace=True)
        #data_sell_tp.drop(['Portfolio Gain'], axis=1, inplace=True) 

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