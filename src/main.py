#!/usr/bin/env python
# coding: utf-8
from timeit import default_timer as timer
start = timer()
import os
import sys
import pytz
from datetime import datetime as dt
import logging
import argparse
from check_market import market_open
from trade.logic import Decide
from trade.openTrades import openTrades
from trade.trade import Trade
from trading import Trading
from trade.order import Order
from send_predictions.telegram_send import telegram_bot
from send_predictions.email_send import send_email, create_html, from_html_to_jpg, make_image
from saveToDB import save_order, save_input
from risk_management import check_stop_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append('./src/assets/')
sys.path.append('./src')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
)


def main(argv):
    """
    Main
    """
    TOKEN = os.environ['telegram_token']
    CHAT_ID = os.environ['telegram_chat_id']
    initial_pip = float(os.environ['initial_pip'])
    html_template_path = "./assets/email/email_template.html"

    tz_MX = pytz.timezone('America/Mexico_City')
    datetime_MX = dt.now(tz_MX)

    hora_now = f'{datetime_MX.strftime("%H:%M:%S")}'

    parser = argparse.ArgumentParser(description='Apollo V 0.1 Beta')

    parser.add_argument('-o', '--order', action='store_true',
                        help='Determine if you want to make an order')
    parser.add_argument('-t', '--time', action='store_true',
                        help='Make order only if market is open')
    parser.add_argument('-m', '--model-version',
                        help='Model version folder simple name')
    parser.add_argument('-i', '--instrument',
                        help='instrument to trade')
    parser.add_argument('-a', '--account',
                        help='suffix of account to trade')
    parser.add_argument('-s', '--save', action='store_true',
                        help='saves the predictions made')

    args = parser.parse_args()
    make_order = args.order or False
    market_sensitive = args.time or False
    model_version = args.model_version
    instrument = args.instrument
    account = args.account
    save_preds = args.save or False
    trading = Trading(model_version, instrument)
    op_buy, op_sell, original_dataset = trading.predict()
    previous_low_bid = str(original_dataset[2]['USD_JPY_lowBid'].iloc[-2].round(3))
    previous_high_ask = str(original_dataset[2]['USD_JPY_highAsk'].iloc[-2].round(3))

    logging.info(f'\nMarket sensitive: {market_sensitive}')
    if market_sensitive and not market_open():
        logging.info('Market Closed')
        open_trades = openTrades(account)
        current_pips = open_trades.get_pips_traded()
        current_trades = open_trades.get_all_trades()
        check_stop_loss(current_trades,account)
        return

    # Hacer decisón para la posición
    decision = Decide(op_buy, op_sell, 100000, direction=0,
                      pips=initial_pip, take_profit=0)
    decision.get_all_pips()
    units = decision.pips * decision.direction * 1000

    # máximo de unidades en riesgo al mismo tiempo

    pip_limit = float(os.environ['pip_limit'])
    open_trades = openTrades(account)
    current_pips = open_trades.get_pips_traded()
    current_trades = open_trades.get_all_trades()


    logging.info(f'Current units: {current_pips}')
    logging.info(f'Max units: {pip_limit}')
    logging.info(f'Units: {units}')

    # si queremos hacer una operación (units puede ser positivo o negativo)
    if units != 0:
        if current_pips < pip_limit:  # vemos si aún podemos hacer operaciones
            # escogemos lo que podamos operar sin pasarnos del límite.
            # el mínimo entre la unidades solicitadas o las disponibles
            units = min(abs(units), pip_limit - current_pips) * \
                decision.direction
            if units == 0.0:  # si encontramos que ya no hay
                decision.decision += '\n*Units limit exceeded. Order not placed.'
        else:  # si ya hemos excedido operaciones
            units = 0.0
            decision.decision += '\n*Units limit exceeded. Order not placed.'

    inv_instrument = 'USD_JPY'
    take_profit = decision.take_profit
    op_buy_new = decision.data_buy
    op_sell_new = decision.data_sell

    logging.info(f'inv_instrument: {inv_instrument}')
    logging.info(f'take_profit: {take_profit}')

    logging.info(f'\n{decision.decision}')
    # Pone orden a precio de mercado
    logging.info(
        f'Units: {units}, inv_instrument: {inv_instrument} , take_profit: {take_profit}\n')

    if make_order and units != 0:
        new_trade = Trade(inv_instrument, units, take_profit=take_profit)
        new_order = Order(new_trade, account)
        new_order.make_market_order()
        previous_low_bid += f' ({new_order.bid_price})'
        previous_high_ask += f' ({new_order.ask_price})'
        end = timer()
        speed_time = end - start
        logging.info('Apollo time to market: ' + str(end - start))
        if save_preds:
            logging.info('\n\n************* Saving predictions in Data Base **************')
            save_order(account,
                       model_version,
                       inv_instrument,
                       new_order,
                       decision.probability)
            logging.info(f'\n\n************* Saving dataset in Data Base **************')
            save_input(account,
                    model_version,
                    hora_now,
                    inv_instrument, 
                    original_dataset,
                    order_id=new_order.i_d)
    else:
        end = timer()
        speed_time = end - start
        logging.info(f'Apollo prediction time: {str(speed_time)} s')
        logging.info(f'\n\n************* Saving dataset in Data Base **************')
        save_input(account, model_version, hora_now, inv_instrument, 
                original_dataset)
    
    logging.info('\n\n ************* Checando trades activos  **************')
    check_stop_loss(current_trades,account)

    logging.info(f'\nPrevious High Ask:{previous_high_ask}')
    logging.info(op_buy_new)
    logging.info(f'\nPrevious Low Bid: {previous_low_bid}')
    logging.info(op_sell_new)

    # send telegram
    _, html_path = create_html(
        [op_buy, op_sell, previous_high_ask, previous_low_bid, f'{model_version}'], html_template_path)
    _, image_name = from_html_to_jpg(html_path)
    logging.info('Sending predictions to Telegram')
    bot = telegram_bot(TOKEN)
    bot.send_message(CHAT_ID, f"Predictions for the hour: {hora_now} ({model_version})")
    bot.send_photo(CHAT_ID, image_name)
    bot.send_message(
        CHAT_ID, f"Best course of action ({model_version}): {decision.decision}\nApollo speed:{str(round(speed_time,3))}s")
    
    
    logging.info(f'Apollo prediction time: {str(speed_time)} s')

if __name__ == "__main__":
    # load settings
    with open("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
