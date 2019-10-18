# coding: utf-8
import numpy as np
import pandas as pd
from .pip_calculator import get_profit, get_loss
import time

class Decide:
    
    def __init__(self, data_buy, data_sell, portfolio, direction=0, pips=0, take_profit=0 , stop_loss=0):
        """
        Makes a decision about the next operation

        Args:
            - data (Dataframe): gett

        """
        self.data_buy = data_buy
        self.data_sell = data_sell
        self.direction = direction
        self.pips = pips
        self.take_profit = take_profit
        self.decision = ''
        self.spread = data_buy.iloc[0]['Open'] - data_sell.iloc[0]['Open']
    
    def get_best_action(self, buy_sell):
        if buy_sell == 'Buy':
            data = self.data_buy
        elif buy_sell == 'Sell':
            data = self.data_sell

        pips = self.pips
        spread = self.spread * pips * 10  #checar la lógica/razón de esto

        data[f'Expected Profit ({pips})'] =  round(get_profit(data['Open'], data["Take Profit"], pips),2)

        print(f'\nSearching for best {buy_sell} strategy(TP):')
        
        for i in reversed(range(len(data)-1)): #vamos checando del valor de TP más alejado al más cercano
            best_action = data.iloc[i]
            probability = best_action['Probability']
            profit = get_profit(best_action['Open'], best_action["Take Profit"], pips)
            # Si la proba es > 0.6, y la ganancia cubre al menos el spread, entonces ese utilizamps
            if probability >= 0.6 and profit-spread > 0:
                return best_action, profit            
        # Si no encontramos entonces tomamos el que tenga la mayor proba y calculamos su TP
        best_action = data.loc[data['Probability'].idxmax()]
        
        
        return best_action, 0


    
    def get_all_pips(self):
        data_buy_tp = self.data_buy
        data_sell_tp = self.data_sell

        # Se toman los TP y SL para Buy y Sell que tengan la mayor utilidad y cubran el spread
        data_buy_tp_aux = data_buy_tp.copy()
        data_buy_sl_aux = data_sell_tp.copy()
        pips = self.pips

        # Se calcula decisión
        best_action_buy, profit_buy = self.get_best_action('Buy')
        best_action_sell, profit_sell = self.get_best_action('Sell')


        print(f'Best Action Buy:\n{best_action_buy}\nBest Action Sell:\n{best_action_sell}')
        if profit_buy > profit_sell:
            self.decision = '\nBuy!\n' + f"Take Profit: ${best_action_buy['Take Profit']}"
            self.decision += f"\nProbability: {round(best_action_buy['Probability']*100,2)}%"
            self.decision += f"\nProfit[{self.pips}]: ${round(profit_buy,2)}"
            self.decision += f'\nSpread[{self.pips}]: ${round(self.spread * self.pips * 10,2)}'
            self.direction = 1
            self.take_profit = str(best_action_buy['Take Profit'])
        elif profit_sell > profit_buy:
            self.decision = '\nSell!\n' + f"Take Profit: ${best_action_sell['Take Profit']}"
            self.decision += f"\nProbability: {round(best_action_sell['Probability']*100,2)}%"
            self.decision += f"\nProfit[{self.pips}]: ${round(profit_sell,2)}"
            self.decision += f'\nSpread[{self.pips}]: ${round(self.spread * self.pips * 10,2)}'
            self.direction = -1
            self.take_profit = str(best_action_sell['Take Profit'])
        else:
            self.decision = '\nNeutral \nBuy '
            self.decision += f"\nBuy Gain: ${round(get_profit(best_action_buy['Open'], best_action_buy['Take Profit'], pips),3)}"
            self.decision += f"\nProbability: {round(best_action_buy['Probability']*100,2)}%"
            self.decision += f"\nLast Best Buy Price{best_action_buy['Take Profit']}"
            self.decision += f"\nSell\nSell Gain: ${round(get_profit(best_action_sell['Open'], best_action_sell['Take Profit'], pips),3)}"
            self.decision += f"\nProbability: {round(best_action_sell['Probability']*100,2)}%"
            self.decision += f"\nLast Best Sell Price{best_action_sell['Take Profit']}"
            self.decision += f'\nSpread: ${round(self.spread * self.pips * 10,2)}'

        

        self.data_buy = data_buy_tp
        self.data_sell = data_sell_tp


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