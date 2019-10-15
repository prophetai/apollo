#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytz
import time
from datetime import datetime as dt
import imgkit

#librerías para mandar correo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from trade.order import Order
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table 



COMMASPACE = ', '

def send_email(subject,fromaddr, toaddr, password, html_file):
    """
    Manda email de tu correo a tu correo
    Args:
        subject (str): Asunto del correo
        body_test (str): Cuerpo del correo
    """

    toaddr = toaddr.split(' ')

    # datetime object with timezone awareness:
    dt.now(tz=pytz.utc)

    # seconds from epoch:
    dt.now(tz=pytz.utc).timestamp()

    # ms from epoch:
    hora_now = int(dt.now(tz=pytz.utc).timestamp() * 1000)
    hora_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    msg = MIMEMultipart()
    msg.preamble = f'Predicciones de la hora {hora_now}'
    msg['From'] = fromaddr
    msg['To'] = COMMASPACE.join(toaddr)
    msg['Subject'] = subject + ' '+str(hora_now)

    msg.attach(MIMEText(html_file, 'html'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def create_html(body_text, html_template_path):
    """
    Populates the html with predictions data
    """
    html_template = open(html_template_path, 'r')
    html_template = html_template.read()
    soup = BeautifulSoup(html_template, features="lxml")    
    
    find_buy_price = soup.find("p", {"id": "previous_price"})
    br = soup.new_tag('br')
    prices = [body_text[-2], body_text[-1]]
    tables = [body_text[0], body_text[1]]

    for i, price in enumerate(soup.select('p.content')):
        price.replace_with(str(prices[i]))
    
    find_buy = soup.find("table", {"id": "buy_table"})
    br = soup.new_tag('br')
    for i, table in enumerate(soup.select('table.dataframe')):
        table.replace_with(BeautifulSoup(tables[i].to_html(), "html.parser"))

    html_path = f"USDJPY_predictions.html"
    with open(html_path, "w") as file:
        file.write(str(soup))

    return soup, html_path

def from_html_to_jpg(html_path):
    config = imgkit.config(wkhtmltoimage='/usr/local/bin/wkhtmltoimage')
    file_name = html_path.split('.html')[0] + '.jpg'
    image = imgkit.from_file(html_path, file_name, config=config)

    return image, file_name


def make_image(df, name):
    ax = plt.subplot(111, frame_on=False) # no visible frame
    #ax.xaxis.set_visible(False)  # hide the x axis
    #ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, df)  # where df is your data frame

    file_name = f'{name}_picture.png'
    plt.savefig(file_name)

    return file_name
