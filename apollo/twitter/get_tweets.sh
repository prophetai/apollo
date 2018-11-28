#!/bin/sh
cd ~/stock_prophets/crons/twitter/
python3 twitter.py -a ~/docs/cuentas_fx.csv -c ~/docs/creds.txt -d True
