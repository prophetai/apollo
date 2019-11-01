#!/usr/bin/env python
# coding: utf-8

from google.cloud import storage
import os

client = storage.Client()

bucket = client.get_bucket('forex_models')

blob = bucket.get_blob('models/variables/variablesHigh.csv')

print(blob)