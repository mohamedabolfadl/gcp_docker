# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""


####################################
#--       IMPORT LIBRARIES       
####################################


import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import talib as ta

import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda
import datetime
import joblib
import yfinance as yf
import itertools
import numerapi
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import alpaca_trade_api as tradeapi


SCORE_TRIPLE_BARRIER = False
WINDOW_LONG = 200
WINDOW_MEDIUM = 100
WINDOW_SHORT = 50
WINDOW = 20
WINDOW_TS = 20

NMR_P_KEY = 'DKVW6H65GQC3ZBXOG2MWYG4H4IIAWLBS'
NMR_S_KEY = 'C2H6GPNF7ZYVNO5JNKEBHSL43A34XIQVWREOS2IPT54B3C5HJLEIPSQAYRXGFPCB'
ACCOUNT_NAME = 'mabolfadl2'

from sklearn.base import TransformerMixin, BaseEstimator


alpaca_pk = 'PK87CSMW3Y3OM94718RU'
alpaca_sk = 'e4JX1Yl5T87tsuE3yV/6tXUEUTbq0CAxUNrRYZ9o'


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import alpaca_trade_api as tradeapi

utils_dir = '11_utils/'

CREDS_JSON_FILE_PATH = utils_dir+'sheets_api.json'

#=====GCS PATHS
BUCKET_NAME = 'abolfadl-stk-live'
PRICES_GCS_PATH = BUCKET_NAME + '/prices/'
PREDS_GCS_PATH = BUCKET_NAME + '/preds/'
MODELS_GCS_PATH = BUCKET_NAME + '/mdls/'
FEATURES_GCS_PATH = BUCKET_NAME + '/features/'
NMR_GCS_PATH = BUCKET_NAME + '/nmr/'


####################################
#--       INPUT VARIABLES       
####################################

data_input_dir = "02_data/input/"
data_intermediate_dir = "02_data/intermediate/"
data_output_dir = "02_data/output/"
models_prod_dir = "03_models/prod/"
models_archive_dir = "03_models/archive/"
data_live_dir = "02_data/live/"
LATEST_PREDICTIONS_FILE = data_live_dir+'predictions/latest_predictions.csv'


GCS_CREDS ={
  "type": "service_account",
  "project_id": "stock-288218",
  "private_key_id": "614350c8d3d4fb606661977272093161a711d162",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCuIm7UODsubgtz\n0d+3tja6m8SpqB6l/G+7iK+Gz+q0MIG7qzFAwgrpCWpDbyQpSEMVto014nMoilis\nNiTLKH0uGcaouO5IxSN0oEIDVSKCIlQT9U7tm/WqLQ6eCgxhiwP9XO8cO9zDw6xm\nsXRreH9puLhH4XpElK4sjHru46ky0LbBdLOXO8rAs/ml90B/Twtv4J+25UieBYNC\n2e36I+ZjCksmE36KSZK5JyrNRi6cyxX+H5nX7Xg82AiqUZQl7XiyknfJ+Z7GC96x\nWeaQGjB4lVAa1G8YaqWeoJmUOfRT8PKPBnYq+qq791q/it7cxn41YTzk5xeIwMH3\nTt0cPgXrAgMBAAECggEALirb5nEYI7myWJ0yqrLpNLV6wR5dLdXNS9Oz8dKrH+Xi\nZ36+WrR3jwxbe5B6bmWFv7p8Gua0cHGpi9L5E1HjSnc0B+Sr7Ggz+8ZHajnGoej3\nEKmPqQiZ6+nxP65bVPs17hUXIg2u/MoiqcFvo9S5Ny7t0MTzlT40JYSDTVXXtKxY\nZsfTdTGY9hIgGfavPKUAt7CXBwn81LIQkRB5qs7vaZ+N4UdvhZP8Jej84BSprskO\nnPavbNo4FNxIvSggznAKtQfFWeBwvCmjynWnyBzmyekznHRbOwzv9Ic+aYDfQ3Ck\nvcxkwAzKUz7V/mJQk6eQwXfAEX4W+5MKyqGzsAVY8QKBgQDWxC4+Ne/c5M3mPke+\nHs1tZEuOkWugyMZmwOPBBDmcswDel9/G8feXUD5D+JyMageL+HDN6Q3yoKBIpDgL\nnbxuALadq2T5/VwcAyAPsHcPSfn0ZI8+jCen2nr1pgmrp4WtwfMkRtky4HYKWxE7\n7oSNIou8rgED4IUhOfjthaenewKBgQDPkTAPbLqYwaZhfZytU8uUvP9KoHZC76DR\nv6CqwAa8GVJ1GI8CGBw1kGSsNJ7htvaGiZYgj6YsVQNmoT6Jaj+JjFVLMi+9jJsL\nx2BqSNh8799LKB991VCRaPWLSQImU0xRj6C807ALh5ian5h/APeGUKAwAnmaQkyy\n0J1K1/OYUQKBgAXTmLWTXQiPJI7kjam4yDX5jJs3ksXv7IIJJaSs6qs3qARh9m49\nTkxKnwXVDto4XjAL83OCqSA1/2M5IVQfWWdamZm5dqnZl9AivZHvZauChEd/GuvF\nfaXuJXvYn1aK2vOV2XTYfzgEIu+w/My1cd8qWsQprFlgk/wMyJYZFC1VAoGAYUYA\nSOpClD4EdCHC4DOp732XAmkqovnb2xA8AmlVfqc7TmcA9hFIfw25MD7EyrDM3YXz\nFjVbweDhZCJixVFGj3Z1rnTJjMItExsPox+aXQqpEXavM3BZfpu6ntjLKhzVYk+2\nG2f1U6HrMWNjuvlABvEX8Qfn5xP1D8PWQvHrVdECgYAz6lSVIwk/V8fB5k4U1RnZ\nGgJOWlzhlPUg2S/rif6ahEd3W3Z9mMmncmJ93lEKlR5XruX9B8vgMqrea9tUgdB+\ni7iujZKwzJg43X3Bsc11hvgyxjTObmqRKcNI/ngxpU4BM8iy5mnIRdZFtbvhXK18\nedfCq6Y9aIMup1eeXJxo7Q==\n-----END PRIVATE KEY-----\n",
  "client_email": "liveprice@stock-288218.iam.gserviceaccount.com",
  "client_id": "103532957703350079927",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/liveprice%40stock-288218.iam.gserviceaccount.com"
}





import gcsfs
def read_csv_gcs(FILEPATH, creds = GCS_CREDS, project = 'stock'):
    '''
    

    Parameters
    ----------
    FILEPATH : string
        Path to file in gcs e.g. 'gs://abolfadl-stk-live/preds/latest_predictions.csv'.
    creds : dictionary
        Dictionary of credentials to access gcs.
        For e.g. 
                {
                  "type": "service_account",
                  "project_id": "stock-288218",
                  "private_key_id": "614350c8d3d4fb606661977272093161a711d162",
                  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCuIm7UODsubgtz\n0d+3tja6m8SpqB6l/G+7iK+Gz+q0MIG7qzFAwgrpCWpDbyQpSEMVto014nMoilis\nNiTLKH0uGcaouO5IxSN0oEIDVSKCIlQT9U7tm/WqLQ6eCgxhiwP9XO8cO9zDw6xm\nsXRreH9puLhH4XpElK4sjHru46ky0LbBdLOXO8rAs/ml90B/Twtv4J+25UieBYNC\n2e36I+ZjCksmE36KSZK5JyrNRi6cyxX+H5nX7Xg82AiqUZQl7XiyknfJ+Z7GC96x\nWeaQGjB4lVAa1G8YaqWeoJmUOfRT8PKPBnYq+qq791q/it7cxn41YTzk5xeIwMH3\nTt0cPgXrAgMBAAECggEALirb5nEYI7myWJ0yqrLpNLV6wR5dLdXNS9Oz8dKrH+Xi\nZ36+WrR3jwxbe5B6bmWFv7p8Gua0cHGpi9L5E1HjSnc0B+Sr7Ggz+8ZHajnGoej3\nEKmPqQiZ6+nxP65bVPs17hUXIg2u/MoiqcFvo9S5Ny7t0MTzlT40JYSDTVXXtKxY\nZsfTdTGY9hIgGfavPKUAt7CXBwn81LIQkRB5qs7vaZ+N4UdvhZP8Jej84BSprskO\nnPavbNo4FNxIvSggznAKtQfFWeBwvCmjynWnyBzmyekznHRbOwzv9Ic+aYDfQ3Ck\nvcxkwAzKUz7V/mJQk6eQwXfAEX4W+5MKyqGzsAVY8QKBgQDWxC4+Ne/c5M3mPke+\nHs1tZEuOkWugyMZmwOPBBDmcswDel9/G8feXUD5D+JyMageL+HDN6Q3yoKBIpDgL\nnbxuALadq2T5/VwcAyAPsHcPSfn0ZI8+jCen2nr1pgmrp4WtwfMkRtky4HYKWxE7\n7oSNIou8rgED4IUhOfjthaenewKBgQDPkTAPbLqYwaZhfZytU8uUvP9KoHZC76DR\nv6CqwAa8GVJ1GI8CGBw1kGSsNJ7htvaGiZYgj6YsVQNmoT6Jaj+JjFVLMi+9jJsL\nx2BqSNh8799LKB991VCRaPWLSQImU0xRj6C807ALh5ian5h/APeGUKAwAnmaQkyy\n0J1K1/OYUQKBgAXTmLWTXQiPJI7kjam4yDX5jJs3ksXv7IIJJaSs6qs3qARh9m49\nTkxKnwXVDto4XjAL83OCqSA1/2M5IVQfWWdamZm5dqnZl9AivZHvZauChEd/GuvF\nfaXuJXvYn1aK2vOV2XTYfzgEIu+w/My1cd8qWsQprFlgk/wMyJYZFC1VAoGAYUYA\nSOpClD4EdCHC4DOp732XAmkqovnb2xA8AmlVfqc7TmcA9hFIfw25MD7EyrDM3YXz\nFjVbweDhZCJixVFGj3Z1rnTJjMItExsPox+aXQqpEXavM3BZfpu6ntjLKhzVYk+2\nG2f1U6HrMWNjuvlABvEX8Qfn5xP1D8PWQvHrVdECgYAz6lSVIwk/V8fB5k4U1RnZ\nGgJOWlzhlPUg2S/rif6ahEd3W3Z9mMmncmJ93lEKlR5XruX9B8vgMqrea9tUgdB+\ni7iujZKwzJg43X3Bsc11hvgyxjTObmqRKcNI/ngxpU4BM8iy5mnIRdZFtbvhXK18\nedfCq6Y9aIMup1eeXJxo7Q==\n-----END PRIVATE KEY-----\n",
                  "client_email": "liveprice@stock-288218.iam.gserviceaccount.com",
                  "client_id": "103532957703350079927",
                  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                  "token_uri": "https://oauth2.googleapis.com/token",
                  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/liveprice%40stock-288218.iam.gserviceaccount.com"
                }
    project : string, optional
        name of the gcp project. The default is 'stock'.

    Returns
    -------
    None.

    '''

    
    
    '''
    fs = gcsfs.GCSFileSystem(project='stock', token = utils_dir+'sheets_api.json')
    fs.ls('abolfadl-stk-live')
    with fs.open('gs://abolfadl-stk-live/preds/latest_predictions.csv', 'rb') as f:
        df = pd.read_csv(f) 
    '''
    fs = gcsfs.GCSFileSystem(project='stock', token = creds)
    with fs.open(FILEPATH, 'rb') as f:
        df = pd.read_csv(f) 
    
    return(df)

def read_parquet_gcs(FILEPATH, creds = GCS_CREDS, project = 'stock'):
    '''
    

    Parameters
    ----------
    FILEPATH : string
        Path to file in gcs e.g. 'gs://abolfadl-stk-live/preds/latest_predictions.csv'.
    creds : dictionary
        Dictionary of credentials to access gcs.
        For e.g. 
                {
                  "type": "service_account",
                  "project_id": "stock-288218",
                  "private_key_id": "614350c8d3d4fb606661977272093161a711d162",
                  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCuIm7UODsubgtz\n0d+3tja6m8SpqB6l/G+7iK+Gz+q0MIG7qzFAwgrpCWpDbyQpSEMVto014nMoilis\nNiTLKH0uGcaouO5IxSN0oEIDVSKCIlQT9U7tm/WqLQ6eCgxhiwP9XO8cO9zDw6xm\nsXRreH9puLhH4XpElK4sjHru46ky0LbBdLOXO8rAs/ml90B/Twtv4J+25UieBYNC\n2e36I+ZjCksmE36KSZK5JyrNRi6cyxX+H5nX7Xg82AiqUZQl7XiyknfJ+Z7GC96x\nWeaQGjB4lVAa1G8YaqWeoJmUOfRT8PKPBnYq+qq791q/it7cxn41YTzk5xeIwMH3\nTt0cPgXrAgMBAAECggEALirb5nEYI7myWJ0yqrLpNLV6wR5dLdXNS9Oz8dKrH+Xi\nZ36+WrR3jwxbe5B6bmWFv7p8Gua0cHGpi9L5E1HjSnc0B+Sr7Ggz+8ZHajnGoej3\nEKmPqQiZ6+nxP65bVPs17hUXIg2u/MoiqcFvo9S5Ny7t0MTzlT40JYSDTVXXtKxY\nZsfTdTGY9hIgGfavPKUAt7CXBwn81LIQkRB5qs7vaZ+N4UdvhZP8Jej84BSprskO\nnPavbNo4FNxIvSggznAKtQfFWeBwvCmjynWnyBzmyekznHRbOwzv9Ic+aYDfQ3Ck\nvcxkwAzKUz7V/mJQk6eQwXfAEX4W+5MKyqGzsAVY8QKBgQDWxC4+Ne/c5M3mPke+\nHs1tZEuOkWugyMZmwOPBBDmcswDel9/G8feXUD5D+JyMageL+HDN6Q3yoKBIpDgL\nnbxuALadq2T5/VwcAyAPsHcPSfn0ZI8+jCen2nr1pgmrp4WtwfMkRtky4HYKWxE7\n7oSNIou8rgED4IUhOfjthaenewKBgQDPkTAPbLqYwaZhfZytU8uUvP9KoHZC76DR\nv6CqwAa8GVJ1GI8CGBw1kGSsNJ7htvaGiZYgj6YsVQNmoT6Jaj+JjFVLMi+9jJsL\nx2BqSNh8799LKB991VCRaPWLSQImU0xRj6C807ALh5ian5h/APeGUKAwAnmaQkyy\n0J1K1/OYUQKBgAXTmLWTXQiPJI7kjam4yDX5jJs3ksXv7IIJJaSs6qs3qARh9m49\nTkxKnwXVDto4XjAL83OCqSA1/2M5IVQfWWdamZm5dqnZl9AivZHvZauChEd/GuvF\nfaXuJXvYn1aK2vOV2XTYfzgEIu+w/My1cd8qWsQprFlgk/wMyJYZFC1VAoGAYUYA\nSOpClD4EdCHC4DOp732XAmkqovnb2xA8AmlVfqc7TmcA9hFIfw25MD7EyrDM3YXz\nFjVbweDhZCJixVFGj3Z1rnTJjMItExsPox+aXQqpEXavM3BZfpu6ntjLKhzVYk+2\nG2f1U6HrMWNjuvlABvEX8Qfn5xP1D8PWQvHrVdECgYAz6lSVIwk/V8fB5k4U1RnZ\nGgJOWlzhlPUg2S/rif6ahEd3W3Z9mMmncmJ93lEKlR5XruX9B8vgMqrea9tUgdB+\ni7iujZKwzJg43X3Bsc11hvgyxjTObmqRKcNI/ngxpU4BM8iy5mnIRdZFtbvhXK18\nedfCq6Y9aIMup1eeXJxo7Q==\n-----END PRIVATE KEY-----\n",
                  "client_email": "liveprice@stock-288218.iam.gserviceaccount.com",
                  "client_id": "103532957703350079927",
                  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                  "token_uri": "https://oauth2.googleapis.com/token",
                  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/liveprice%40stock-288218.iam.gserviceaccount.com"
                }
    project : string, optional
        name of the gcp project. The default is 'stock'.

    Returns
    -------
    None.

    '''

    
    
    '''
    fs = gcsfs.GCSFileSystem(project='stock', token = utils_dir+'sheets_api.json')
    fs.ls('abolfadl-stk-live')
    with fs.open('gs://abolfadl-stk-live/preds/latest_predictions.csv', 'rb') as f:
        df = pd.read_csv(f) 
    '''
    fs = gcsfs.GCSFileSystem(project='stock', token = creds)
    with fs.open(FILEPATH, 'rb') as f:
        df = pd.read_parquet(f) 
    
    return(df)


    



def write_csv_gcs(df, FILEPATH):
    '''
    Writes a dataframe to GCS with path FILEPATH on the GCS

    '''

    try:
        client = storage.Client()
        bucket = client.get_bucket('abolfadl-stk-live')
        bucket.blob(FILEPATH).upload_from_string(df.to_csv(), 'text/csv')
        successful = True
    except:
        successful = False

    return successful

def write_parquet_gcs_functions(df, FILEPATH):
    '''
    Writes a dataframe to GCS with path FILEPATH on the GCS. Works from functions since no need for authentication

    '''

    try:
        df.to_parquet(FILEPATH,allow_truncated_timestamps=True)
        successful = True
    except:
        successful = False

    return successful

def write_parquet_gcs_local(df, FILEPATH, JSON_FILE_PATH):
    '''
    Writes a dataframe to GCS with path FILEPATH on the GCS

    '''

    try:
        #-- Dump temp file
        df.to_parquet('tmp.parquet',allow_truncated_timestamps=True)
        storage_client = storage.Client.from_service_account_json(JSON_FILE_PATH)
        bucket = storage_client.get_bucket('abolfadl-stk-live')

        #== Upload to latest predictions
        blob = bucket.blob(FILEPATH)
        blob.upload_from_filename('tmp.parquet')
        successful = True
    except:
        successful = False

    return successful




def getLivePrices():
    '''
    

    Returns
    -------
    df_clean : dataframe
        Connects to google sheets and get dataframe with live stock prices with ~2086 rows.
        
        Columns:
            ['ticker', 'price', 'priceopen', 'volume', 'high', 'low', 'timestamp']
        
          ticker   price  priceopen    volume    high     low timestamp
        0    AIR   20.35      20.37     86417   20.67   20.19          
        1    PNW   73.98      71.54    476970   74.05   71.50          
        2    AAN   57.17      57.18    311909   57.42   55.55          
        3    ABT  109.45     106.45   3393666  109.48  105.88          
        4    AMD   90.42      94.01  42748816   94.28   88.74   

    '''
    # import gspread
    # from oauth2client.service_account import ServiceAccountCredentials
    # utils_dir = '11_utils/'
    
    
    # use creds to create a client to interact with the Google Drive API
    #scope = ['https://spreadsheets.google.com/feeds']
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(utils_dir+'sheets_api.json', scope)
    client = gspread.authorize(creds)
    
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    sheet = client.open("live_prices").sheet1
    data = sheet.get_all_records()  # Get a list of all records
    
    
    
    df = pd.DataFrame(data)
    df.rename(columns = {'priceopen':'open'}, inplace = True)
    
    
    missing_prices = len(df.loc[df.price == '#N/A',:])/len(df)
    missing_open = len(df.loc[df.open == '#N/A',:])/len(df)
    missing_high = len(df.loc[df.high == '#N/A',:])/len(df)
    missing_low = len(df.loc[df.low == '#N/A',:])/len(df)
    missing_volume = len(df.loc[df.volume == '#N/A',:])/len(df)
    
    
    
    #===Clean df
    
    
    df_clean = df.loc[(df.price != '#N/A') & (df.open != '#N/A') &
                      (df.high != '#N/A') & (df.low != '#N/A') &
                      (df.volume != '#N/A'),:].reset_index(drop = True)
    
    fl_cols = ['price', 'open', 'high', 'low', 'volume']
    
    
    for f in fl_cols:
        #df_clean[f] = df_clean[f].astype(float)
        df_clean[f] = pd.to_numeric(df_clean[f] , errors = 'ignore')
    
    
    
    return df_clean



def getTechnicalFeatures(df_inp, symbol_vec, WINDOW_TS=50):
    '''
    

    Parameters
    ----------
    df_orig : dataframe
        Must have the following columns: close, high, close, open, volume.

    Returns
    -------
    dataframe with all features.

    '''

    
    for j, symbol in enumerate(symbol_vec):
        #-- symbol = 'YRCW'
        if j%100==0:
            print(symbol+' '+str(j))
            
        df_orig = df_inp[(df_inp.symbol == symbol)]
    
        
        df_feat = pd.DataFrame(index = df_orig.index)
        df_feat = df_feat.sort_index()


        close = df_orig.close
        high  = df_orig.high
        low = df_orig.low
        open_ = df_orig.open
        volume = df_orig.volume
        
        
        
       ######################################
        #--  Overlap
        ######################################
        
        #-- BBANDS
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df_feat = pd.DataFrame({'feat_bb_lower':close - lowerband, 'feat_bb_upper':close - upperband})
        
        #-- DEMA
        df_feat['feat_dema'] = close  - ta.DEMA(close, timeperiod=30)
        
        
        #-- EMA
        df_feat['feat_ema_50']= close - ta.EMA(close, timeperiod=50)
        df_feat['feat_ema_100']= close - ta.EMA(close, timeperiod=100)
        df_feat['feat_ema_200']= close - ta.EMA(close, timeperiod=200)
        
        
        #-- Hilbert Transform - Instantaneous Trendline
        df_feat['feat_ht_hilbert']= ta.HT_TRENDLINE(close) 
        df_feat['feat_ht_hilbert']=df_feat['feat_ht_hilbert']- df_feat['feat_ht_hilbert'].shift(1)
        
        #-- KAMA
        df_feat['feat_kama']= close - ta.KAMA(close, timeperiod=30)
        
        #-- MA
        df_feat['feat_ma_50']= close - ta.MA(close, timeperiod=50)
        df_feat['feat_ma_100']= close - ta.MA(close, timeperiod=100)
        df_feat['feat_ma_200']= close - ta.MA(close, timeperiod=200)
        
        #-- MAMA
        # mama, fama = ta.MAMA(close, fastlimit=10, slowlimit=50)
        # df_feat['feat_mama']=mama
        # df_feat['feat_fama']=fama
        
        #-- MIDPOINT
        df_feat['feat_midpoint'] = close - ta.MIDPOINT(close, timeperiod=14)
        
        #-- MIDPRICE
        df_feat['feat_midprice']= close - ta.MIDPRICE(high, low, timeperiod=14)
        
        #-- SAR
        #df_feat['feat_sar'] = ta.SAR(high, low, acceleration=0, maximum=0)
        
        #-- SAR EXT
        df_feat['feat_sarext']= ta.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
        
        
        #-- SMA
        df_feat['feat_sma_50']= close - ta.SMA(close, timeperiod=50)
        df_feat['feat_sma_100']= close - ta.SMA(close, timeperiod=100)
        df_feat['feat_sma_200']= close - ta.SMA(close, timeperiod=200)
        
        #-- T3
        df_feat['feat_t3']= close - ta.T3(close, timeperiod=5, vfactor=0)
        
        #-- TEMA
        df_feat['feat_tema']= close - ta.TEMA(close, timeperiod=30)
        
        #-- TRIMA
        df_feat['feat_trima']= close - ta.TRIMA(close, timeperiod=30)
        
        #-- WMA
        df_feat['feat_wma_50']= close - ta.WMA(close, timeperiod=50)
        df_feat['feat_wma_100']= close - ta.WMA(close, timeperiod=100)
        df_feat['feat_wma_200']= close - ta.WMA(close, timeperiod=200)
        
        #####################################
        #-- Moments
        ####################################
        df_feat['mean_close'] = (df_orig.close - df_orig.close.rolling(WINDOW_TS).mean()) / df_orig.close
        df_feat['std_close'] = df_orig.close.rolling(WINDOW_TS).std()
        df_feat['skew_close'] = df_orig.close.rolling(WINDOW_TS).skew()
        df_feat['kurt_close'] = df_orig.close.rolling(WINDOW_TS).kurt()
    
        
        ######################################
        #--  Momentum
        ######################################
        
        #-- ADX
        #df_feat['feat_adx'] = ta.ADX(high, low, close, timeperiod=14)
        
        #-- ADXR
        df_feat['feat_adxr']= ta.ADXR(high, low, close, timeperiod=14)
        
        #-- APO
        df_feat['feat_apo']= ta.APO(close, fastperiod=12, slowperiod=26, matype=0)
        
        #--AROON
        aroondown, aroonup = ta.AROON(high, low, timeperiod=14)
        df_feat['feat_aroon_up'] = close - aroonup
        df_feat['feat_aroon_down'] = close - aroondown
        
        #-- AROONOSC
        df_feat['feat_aroon_osc']= ta.AROONOSC(high, low, timeperiod=14)
        
        #-- BOP
        df_feat['feat_aroon_bop'] = ta.BOP(open_, high, low, close)
        
        #-- CCI
        df_feat['feat_cci'] = ta.CCI(high, low, close, timeperiod=14)
        
        #-- CMO
        df_feat['feat_cmo'] = ta.CMO(close, timeperiod=14)
        
        #-- DX
        df_feat['feat_dx']= ta.DX(high, low, close, timeperiod=14)
        
        #-- MACD
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df_feat['feat_macd'] = macd
        df_feat['feat_macd_signal'] = macdsignal
        df_feat['feat_macd_hist'] = macdhist
        
        
        #-- MACDFIX
        macd, macdsignal, macdhist = ta.MACDFIX(close, signalperiod=9)
        df_feat['feat_macdfix'] = macd
        df_feat['feat_macd_signalfx'] = macdsignal
        df_feat['feat_macd_histfx'] = macdhist
        
        #-- MFI
        df_feat['feat_mfi']= ta.MFI(high, low, close, volume, timeperiod=14)
        
        #-- MINUS DI
        df_feat['feat_minus_di']= ta.MINUS_DI(high, low, close, timeperiod=14)
        
        #-- MINUS DM
        #df_feat['feat_minus_dm']= ta.MINUS_DM(high, low, close, timeperiod=14)
        
        #-- MOM
        df_feat['feat_mom']= ta.MOM(close, timeperiod=10)
        
        #-- PLUS DI
        df_feat['feat_plus_di']= ta.PLUS_DI(high, low, close, timeperiod=14)
        
        #-- PLUS DM
        #df_feat['feat_plus_dm']= ta.PLUS_DM(high, low, close, timeperiod=14)
        
        #-- PPO
        df_feat['feat_ppo'] = ta.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        
        
        #-- ROC
        df_feat['feat_roc'] = ta.ROC(close, timeperiod=10)
        
        #-- ROCP
        df_feat['feat_rocp'] = ta.ROCP(close, timeperiod=10)
        
        #-- ROCR
        df_feat['feat_rocr'] = ta.ROCR100(close, timeperiod=10)
        
        #-- RSI
        df_feat['feat_rsi']= ta.RSI(close, timeperiod=14)
        
        #--STOCH
        slowk, slowd = ta.STOCH(high, low, close)
        fastk, fastd = ta.STOCHF(high, low, close)
        df_feat['feat_stoch_slow'] = slowk
        df_feat['feat_stoch_fast'] = fastk
        
        #-- STOCHF
        fastk, fastd = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        df_feat['feat_stoch_fast_slow'] = slowk
        df_feat['feat_stoch_fast_fast'] = fastk
        
        #-- STOCHRSI
        fastk, fastd = ta.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df_feat['feat_stoch_rsi_slow'] = slowk
        df_feat['feat_stoch_rsi_fast'] = fastk
        
        
        #-- TRIX
        df_feat['feat_trix'] = ta.TRIX(close, timeperiod=30)
        
        #-- ULTOSC
        df_feat['feat_ultosc']= ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        #-- WILLR
        df_feat['feat_willr']= ta.WILLR(high, low, close, timeperiod=14)
        
        
        
        #########################
        #-- Volume
        #########################
        
        #-- AD
        #df_feat['feat_ad'] = ta.AD(high, low, close, volume)
        
        #-- ADOSC
        df_feat['feat_adosc']  = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        #-- OBV
        #df_feat['feat_obv'] = ta.OBV(close, volume)
        
        
        ##########################
        #-- Volatility 
        ##########################
        
        #-- ATR
        df_feat['feat_atr'] = ta.ATR(high, low, close, timeperiod=14)
        df_feat['feat_atr'] = df_feat['feat_atr']  - df_feat['feat_atr'].shift(1)
    
        #-- NATR 
        df_feat['feat_natr'] = ta.NATR(high, low, close, timeperiod=14)
        df_feat['feat_natr'] = df_feat['feat_natr']  - df_feat['feat_natr'].shift(1)
        
        #-- TRANGE
        df_feat['feat_trange'] = ta.TRANGE(high, low, close)
        df_feat['feat_trange'] = df_feat['feat_trange']  - df_feat['feat_trange'].shift(1)
    
        
        
        ###########################3
        #-- Price
        ###########################3
        
        #-- avg price
        df_feat['feat_avgprice']= close  - ta.AVGPRICE(open_, high, low, close)
        #-- MEDPRICE
        df_feat['feat_medprice']= close  - ta.MEDPRICE(high, low)
        #-- TYPPRICE 
        df_feat['feat_typprice']= close  - ta.TYPPRICE(high, low, close)
        #-- WCLPRICE
        df_feat['feat_wclprice']= close  - ta.WCLPRICE(high, low, close)
        
        ###########################3
        #-- Cycle
        ###########################3
        
        #-- HT_DCPERIOD 
        df_feat['feat_ht_dcperiod'] = ta.HT_DCPERIOD(close)
        
        #-- HT_DCPHASE 
        df_feat['feat_dcphase'] = ta.HT_DCPHASE(close)
        
        #-- HT_PHASOR
        inphase, quadrature = ta.HT_PHASOR(close)
        df_feat['feat_inphase'] = inphase
        df_feat['feat_quadrature'] = quadrature
        
        #--HT_SINE 
        sine, leadsine = ta.HT_SINE(close)
        df_feat['feat_sine'] = sine
        df_feat['feat_leadsine'] = leadsine
        
        #-- HT_TRENDMODE 
        df_feat['feat_ht_trendmode'] = ta.HT_TRENDMODE(close)
        
        
        ###########################3
        #-- Candles
        ###########################3
        
        fncs = ta.get_functions()
        cdl_fncs = [f for f in ta.get_functions() if 'CDL' in f]
        
        #[df_tmp[f] = eval('ta.'+f)(open_,high, low, close) for f in cdl_fncs]
        
        i=0
        while i< len(cdl_fncs):
            #print(str(i))
            if 'feat_'+cdl_fncs[i].lower() not in ['feat_cdl3starsinsouth','feat_cdlconcealbabyswall','feat_cdlmathold','feat_cdlkickingbylength','feat_cdlkicking','feat_cdlabandonedbaby','feat_cdlbreakaway','feat_cdlidentical3crows','feat_cdl3blackcrows','feat_cdlrisefall3methods','feat_cdlupsidegap2crows']:
                try:
                    df_feat['feat_'+cdl_fncs[i].lower()] = eval('ta.'+cdl_fncs[i])(open_,high, low, close)
                except:
                    print('Failed for '+str(i)+' '+cdl_fncs[i])
            
            i=i+1
                #-- Label symbol
        df_feat['symbol'] = symbol
    
        
        if j==0:
            df_feat_full = df_feat
        else:
            df_feat_full = df_feat_full.append(df_feat)

    return(df_feat_full)



    
    
credentials = ServiceAccountCredentials.from_json_keyfile_name(utils_dir+'sheets_api.json')
storage_client = storage.Client.from_service_account_json(utils_dir+'sheets_api.json')
bucket = storage_client.get_bucket('abolfadl-stk-live')

#== Upload to latest predictions
FILEPATH = 'dockertest/sucess.csv'
blob = bucket.blob(FILEPATH)

df = pd.DataFrame({'t':[1,2,3,4]})
try:
        client = storage.Client()
        bucket = client.get_bucket('abolfadl-stk-live')
        bucket.blob(FILEPATH).upload_from_string(df.to_csv(), 'text/csv')
        successful = True
except:
        successful = False

