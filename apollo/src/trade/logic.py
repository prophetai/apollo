# coding: utf-8
import numpy as np
import pandas as pd
from .pip_calculator import get_profit, get_loss
import time
import logging

class Decide:
    
    def __init__(self, data_buy, data_sell, portfolio, direction=0, pips=0, take_profit=0):
        """
        Makes a decision about the next operation

        Args:
            - data_buy (Dataframe): the buying dataframe
            - data_sell (Dataframe): the selling dataframe
            - portfolio (double): current value of portfolio
            - direction (int)[-1,1]: direction of the trade, buy(1) or sell (-1)
            - pips (int): units to trade
            - take_profit(str): expected price to take profit from trade

        """
        self.data_buy = data_buy
        self.data_sell = data_sell
        self.direction = direction
        self.pips = pips
        self.take_profit = take_profit
        self.decision = ''
        self.spread = data_buy.iloc[0]['Open'] - data_sell.iloc[0]['Open']
        self.probability = np.nan
        self.stop_loss = np.nan
        self.decision_level = np.nan
    
    def get_best_action(self, buy_sell):
        """

        Args:
            - buy_sell(str): option for buy or sell best action
        """
        if buy_sell == 'Buy':
            data = self.data_buy
        elif buy_sell == 'Sell':
            data = self.data_sell

        pips = self.pips
        spread = self.spread * pips * 10  #checar la lógica/razón de esto

        data[f'Expected Profit ({pips})'] = round(get_profit(data['Open'], data["Take Profit"], pips),2)

        logging.info(f'\nSearching for best {buy_sell} strategy(TP):')

        max_profit = 0
        probability = 0
        best_action = None
        for i in range(len(data)-1):
            spread = self.spread * pips * 10  #checar la lógica/razón de esto
            probability_cand = data.iloc[i]['Probability']
            profit_cand = get_profit(data.iloc[i]['Open'], data.iloc[i]["Take Profit"], pips)
            # Si la proba es > 0.7, y la ganancia cubre al menos el spread, entonces ese utilizamos
            if probability_cand >= 0.65:
                logging.info('\nNuevo candidato:')
                logging.info(f'Best_action:\n{data.iloc[i]}')
                if probability_cand >= 0.7:
                    profit_cand *= 2
                    spread *= 2
                if profit_cand/spread >= 1.6 and profit_cand > max_profit:
                    logging.info('\nNuevo ganador:')
                    logging.info(f'Probabilidad:{probability_cand}')
                    logging.info(f'Profit: ${profit_cand}')
                    probability = probability_cand 
                    max_profit = profit_cand
                    best_action = data.iloc[i]
        
        if best_action is None:
            best_action = data.loc[data['Probability'].idxmax()]
        
        return best_action, probability


    
    def get_all_pips(self):
        data_buy_tp = self.data_buy
        data_sell_tp = self.data_sell

        # Se toman los TP y SL para Buy y Sell que tengan la mayor utilidad y cubran el spread
        pips = self.pips

        # Se calcula decisión
        best_action_buy, probability_buy = self.get_best_action('Buy')
        best_action_sell, probability_sell = self.get_best_action('Sell')


        logging.info(f'Best Action Buy:\n{best_action_buy}\nBest Action Sell:\n{best_action_sell}')
        if probability_buy > probability_sell:
            if best_action_buy['Probability'] >= 0.8:
                self.pips *= 2
            self.decision = '\nBuy!\n' + f"Take Profit: ${best_action_buy['Take Profit']}"
            self.decision += f"\nProbability: {round(best_action_buy['Probability']*100,2)}%"
            self.decision += f"\nProfit[{self.pips}]: ${round(best_action_buy.iloc[-1],2)}"
            self.decision += f'\nSpread[{self.pips}]: ${round(self.spread * self.pips * 10,2)}'
            self.direction = 1
            self.take_profit = str(best_action_buy['Take Profit'])
            self.probability = probability_buy
            self.decision_level = best_action_buy.name
            
        elif probability_sell > probability_buy:
            if best_action_sell['Probability'] >= 0.8:
                self.pips *= 2
            self.decision = '\nSell!\n' + f"Take Profit: ${best_action_sell['Take Profit']}"
            self.decision += f"\nProbability: {round(best_action_sell['Probability']*100,2)}%"
            self.decision += f"\nProfit[{self.pips}]: ${round(best_action_sell.iloc[-1],2)}"
            self.decision += f'\nSpread[{self.pips}]: ${round(self.spread * self.pips * 10,2)}'
            self.direction = -1
            self.take_profit = str(best_action_sell['Take Profit'])
            self.probability = probability_sell
            self.decision_level = best_action_sell.name
            
        else:
            self.decision = '\nNeutral \n\nBuy:'
            self.decision += f"\nBuy Gain: ${round(get_profit(best_action_buy['Open'], best_action_buy['Take Profit'], pips),2)}"
            self.decision += f"\nProbability: {round(best_action_buy['Probability']*100,2)}%"
            self.decision += f"\nLast Best Buy Price: ${best_action_buy['Take Profit']}"

            self.decision += f"\n\nSell:\nSell Gain: ${round(get_profit(best_action_sell['Open'], best_action_sell['Take Profit'], pips),2)}"
            self.decision += f"\nProbability: {round(best_action_sell['Probability']*100,2)}%"
            self.decision += f"\nLast Best Sell Price: ${best_action_sell['Take Profit']}"
            
            self.decision += f'\n\nSpread: ${round(self.spread * self.pips * 10,2)}'

        

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
    decision = Decide(df_buy, df_sell, portfolio, direction=0, pips=0, take_profit=0    )
    decision.get_all_pips()
    logging.info(f'Buy Take Profit:\n{decision.data_buy}\n')
    logging.info(f'Sell Take Profit:\n{decision.data_sell}\n')
    logging.info(f'\nDecision: {decision.decision}')