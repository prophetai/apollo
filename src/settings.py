# settings.py
import os
from os.path import join, dirname
from dotenv import load_dotenv

DEBUG = False
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
