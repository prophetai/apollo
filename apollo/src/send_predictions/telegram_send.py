import os
import time
import logging
import telegram
from datetime import datetime as dt

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.DEBUG)

class telegram_bot():
    """
    Class that defines a telegram bot
    """

    def __init__(self, token):
        """Creates a telegram bot from the bot's token.

        Args:
            token (string): Bot's token id
        """
        self.token = token
        self.bot = telegram.Bot(token=token)

    def send_message(self,chat_id, message):
        self.bot.send_message(chat_id=chat_id, text=message)

    def send_file(self,chat_id, file_path):
        self.bot.send_document(chat_id=chat_id, document=open(file_path, 'rb'))

    def send_photo(self,chat_id, photo_path):
        self.bot.send_photo(chat_id=chat_id, photo=open(photo_path, 'rb'))


def main():
    TOKEN = os.environ['telegram_token']
    CHAT_ID = os.environ['telegram_chat_id']
    bot = telegram_bot(TOKEN)

    hora_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )#- dt.timedelta(hours=6))
    bot.send_message(CHAT_ID, f"Predicciones de la hora {hora_now}")
    bot.send_file(CHAT_ID,'./src/assets/email/email_template.html')