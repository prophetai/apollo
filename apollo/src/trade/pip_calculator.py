# coding: utf-8

def get_profit(open_price, tp_price, lots):
        pip_dif = abs(tp_price - open_price)
        profit_yen = pip_dif * lots * 1000
        profit_dlls = round(profit_yen / tp_price,3)

        return profit_dlls

def get_loss(open_price, sl_price, lots):
    pip_dif = sl_price - open_price
    loss_yen = pip_dif * lots * 1000
    loss_dlls = loss_yen / sl_price

    return loss_dlls

def get_loss_sell(open_price, sl_price, lots):
    pip_dif = sl_price - open_price
    loss_yen = pip_dif * lots * 1000
    loss_dlls = loss_yen / sl_price

    return loss_dlls

if __name__ == '__main__':
    open_price = 109.562
    close_price = 109.662
    lots = 1
    print(get_profit(open_price, close_price, lots))