#!/bin/sh
cd ~/stock_prophets/crons/stocks/
python3 tickers.py -t ~/docs/tickers.csv -c ~/docs/creds.txt
