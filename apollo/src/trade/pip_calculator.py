# coding: utf-8

def get_profit(open_price, close_price, lots):
        pip_dif = close_price - open_price

        profit_yen = pip_dif * lots * 1000

        profit_dlls = profit_yen / close_price

        return profit_dlls

if __name__ == '__main__':
    open_price = 109.562
    close_price = 109.678
    lots = 1
    print(get_profit(open_price, close_price, lots))