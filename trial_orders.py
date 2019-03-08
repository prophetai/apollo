from order import Order

units = 1
inv_instrument = "USD_JPY"
stop_loss_price = "110.1"
take_profit_price = "111.5"
# Pone orden a precio de mercado
new_order = Order(inv_instrument, take_profit_price, stop_loss_price)
new_order.make_market_order(units)
