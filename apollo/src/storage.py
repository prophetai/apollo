#!/usr/bin/env python
# coding: utf-8

import os
import logging
import cloudstorage as gcs
from google.appengine.api import app_identity

class Storage:
    
    def __init__(self,bucket_name):
        """
        Storage manager 

        Args:
            - bucket_name (str): Bucket where files are stored
        """
        self.bucket_name = bucket_name

    def get(self):
        bucket_name = os.environ.get(self.bucket_name,
                                    app_identity.get_default_gcs_bucket_name())

        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Demo GCS Application running from Version: '
                            + os.environ['CURRENT_VERSION_ID'] + '\n')
        self.response.write('Using bucket name: ' + bucket_name + '\n\n')

    
    def read_file(self, filename):
        self.response.write('Reading the full file contents:\n')

        gcs_file = gcs.open(filename)
        contents = gcs_file.read()
        gcs_file.close()
        self.response.write(contents)
    
    def create_file(self, filename):
        """Create a file.

        The retry_params specified in the open call will override the default
        retry params for this particular file handle.

        Args:
            filename: filename.
        """
        self.response.write('Creating file %s\n' % filename)

        write_retry_params = gcs.RetryParams(backoff_factor=1.1)
        gcs_file = gcs.open(filename,
                            'w',
                            content_type='text/plain',
                            options={'x-goog-meta-foo': 'foo',
                                    'x-goog-meta-bar': 'bar'},
                            retry_params=write_retry_params)
        gcs_file.write('abcde\n')
        gcs_file.write('f'*1024*4 + '\n')
        gcs_file.close()
        self.tmp_filenames_to_clean_up.append(filename)

if __name__ == '__main__':
    
    df_buy = pd.DataFrame(data_buy)
    df_sell = pd.DataFrame(data_sell)  
    portfolio = 100
    decision = Decide(df_buy, df_sell, portfolio, direction=0, magnitude=0, take_profit=0 , stop_loss=0)
    decision.get_all_pips()
    print(f'Buy Take Profit:\n{decision.data_buy_tp}\n\nBuy Stop Loss:\n{decision.data_buy_sl}')
    print(f'Sell Take Profit:\n{decision.data_sell_tp}\n\nSell Stop Loss:\n{decision.data_sell_sl}')
    print(f'\nDecision: {decision.decision}')