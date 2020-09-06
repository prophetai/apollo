#!/usr/bin/env python
# coding: utf-8
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
from saveToDB import save_order

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
    html_template_path = "./src/assets/email/email_template.html"

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
    previous_low_bid = original_dataset['USD_JPY_lowBid'].iloc[-2].round(3)
    previous_high_ask = original_dataset['USD_JPY_highAsk'].iloc[-2].round(3)

    conn_data = {
        'db_user': os.environ['POSTGRES_USER'],
        'db_pwd': os.environ['POSTGRES_PASSWORD'],
        'db_host': os.environ['POSTGRES_HOST'],
        'db_name': os.environ['db_name']
    }

    logging.info(f'\nMarket sensitive: {market_sensitive}')
    if market_sensitive and not market_open():
        logging.info('Market Closed')
        return

# Hacer decisón para la posición
    decision = Decide(op_buy, op_sell, 100000, direction=0,
                      pips=initial_pip, take_profit=0)
    decision.get_all_pips()
    units = decision.pips * decision.direction * 1000

    # máximo de unidades en riesgo al mismo tiempo

    pip_limit = float(os.environ['pip_limit'])
    open_trades = openTrades(account)
    current_pips = open_trades.number_trades()

    print(f'Current units: {current_pips}')
    print(f'Max units: {pip_limit}')
    print(f'Units: {units}')

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

    print(f'inv_instrument: {inv_instrument}')
    print(f'take_profit: {take_profit}')

    logging.info(f'\n{decision.decision}')
    # Pone orden a precio de mercado
    logging.info(
        f'Units: {units}, inv_instrument: {inv_instrument} , take_profit: {take_profit}\n')

    if make_order and units != 0:
        new_trade = Trade(inv_instrument, units, take_profit=take_profit)
        new_order = Order(new_trade, account)
        new_order.make_market_order()
        previous_low_bid = new_order.bid_price
        previous_high_ask = new_order.ask_price
        
        if save_preds:
            logging.info('Saving predictions in Data Base')
            save_order(account,
                       model_version,
                       inv_instrument,
                       new_order,
                       decision.probability,
                       conn_data)

    
    print(f'\nPrevious High Ask:{previous_high_ask}')
    print(op_buy_new)
    print(f'\nPrevious Low Bid: {previous_low_bid}')
    print(op_sell_new)

    # send telegram
    _, html_path = create_html(
        [op_buy, op_sell, previous_high_ask, previous_low_bid, f'{model_version}'], html_template_path)
    _, image_name = from_html_to_jpg(html_path)
    logging.info('Se mandan predicciones a Telegram')
    bot = telegram_bot(TOKEN)
    if not make_order:
        bot.send_message(CHAT_ID, f"TEST!!!!!")
    bot.send_message(CHAT_ID, f"Predictions for the hour: {hora_now} ({model_version})")
    bot.send_photo(CHAT_ID, image_name)
    bot.send_message(
        CHAT_ID, f"Best course of action ({model_version}): {decision.decision}")


if __name__ == "__main__":
    # load settings
    with open("src/settings.py", "r") as file:
        exec(file.read())

    main(sys.argv)
