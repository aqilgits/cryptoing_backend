import os
import requests
import json
import pandas as pd
from pycaret.regression import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float
from flask_marshmallow import Marshmallow
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import matplotlib.pyplot as plt

# endpoint = 'https://min-api.cryptocompare.com/data/histoday'
# res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500')
# hist = pd.DataFrame(json.loads(res.content)['Data'])
# hist = hist.set_index('time')
# hist.index = pd.to_datetime(hist.index, unit='s')
# target_col = 'close'
# hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
# # print(hist)

# prediction_day = 1

# hist['Future_price'] = hist[['close']]
# hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
# # print(hists)

# df = hists.copy()
# X = np.array(df[df.columns])
# X = X[:len(hists)-prediction_day]
# y = np.array(df['Future_price'])
# y = y[:-prediction_day]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)

# train_data = pd.DataFrame(x_train, columns = df.columns)
# test_data = pd.DataFrame(x_test, columns = df.columns)

# regression_setup = setup(data=train_data, target= 'Future_price', session_id = 123, use_gpu = True)
# best_model = compare_models(sort='r2')
# model = create_model('lar')
# plot_model(model, plot = 'error')
# save_model(model,'Final prediction model BTC')


index = 0
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)
train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)
test_data.drop(["Future_price"], axis='columns', inplace=True)

prevdays =[31,62,93]
avghigh = 0
avglow = 0
avgvolumefrom = 0
avgvolumeto = 0
avgopen = 0
for x in range (1,30):
    for y in prevdays:
        # print(test_data.iloc[((len(test_data)-1)+x)-y])
        high_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['high']
        # print(high_ofprevMonth)
        avghigh = (avghigh + high_ofprevMonth)
        low_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['low']
        avglow = (avglow + low_ofprevMonth)
        vf_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumefrom']
        avgvolumefrom = (avgvolumefrom + vf_ofprevMonth)
        vt_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumeto']
        avgvolumeto = (avgvolumeto + vt_ofprevMonth)
        open_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['open']
        avgopen = (avgopen + open_ofprevMonth)
    avghigh = avghigh/len(prevdays)
    # print(avghigh)
    avglow = avglow/len(prevdays)
    avgvolumefrom = avgvolumefrom/len(prevdays)
    avgvolumeto = avgvolumeto/len(prevdays)
    avgopen = avgopen/len(prevdays)
    test_data = test_data.append({'high' : avghigh, 'low' : avglow, 'volumefrom' : avgvolumefrom, 'volumeto' : avgvolumeto, 'open' : avgopen},ignore_index = True)

saved_final_prediction = load_model('Final prediction modelBTC')

unseen_predictions = predict_model(saved_final_prediction, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
unseen_predictions = unseen_predictions.replace(np.nan, 0)
print(unseen_predictions.tail(40))
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_price'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_label'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# print(crypto)

index = 0
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=ETH&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)
train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)
test_data.drop(["Future_price"], axis='columns', inplace=True)

prevdays =[31,62,93]
avghigh = 0
avglow = 0
avgvolumefrom = 0
avgvolumeto = 0
avgopen = 0
for x in range (1,30):
    for y in prevdays:
        # print(test_data.iloc[((len(test_data)-1)+x)-y])
        high_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['high']
        # print(high_ofprevMonth)
        avghigh = (avghigh + high_ofprevMonth)
        low_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['low']
        avglow = (avglow + low_ofprevMonth)
        vf_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumefrom']
        avgvolumefrom = (avgvolumefrom + vf_ofprevMonth)
        vt_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumeto']
        avgvolumeto = (avgvolumeto + vt_ofprevMonth)
        open_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['open']
        avgopen = (avgopen + open_ofprevMonth)
    avghigh = avghigh/len(prevdays)
    # print(avghigh)
    avglow = avglow/len(prevdays)
    avgvolumefrom = avgvolumefrom/len(prevdays)
    avgvolumeto = avgvolumeto/len(prevdays)
    avgopen = avgopen/len(prevdays)
    test_data = test_data.append({'high' : avghigh, 'low' : avglow, 'volumefrom' : avgvolumefrom, 'volumeto' : avgvolumeto, 'open' : avgopen},ignore_index = True)

saved_final_prediction = load_model('Final prediction modelETH')

unseen_predictions = predict_model(saved_final_prediction, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
unseen_predictions = unseen_predictions.replace(np.nan, 0)
print(unseen_predictions.tail(40))
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_price'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_label'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# print(crypto)

index = 0
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=XRP&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)
train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)
test_data.drop(["Future_price"], axis='columns', inplace=True)

prevdays =[31,62,93]
avghigh = 0
avglow = 0
avgvolumefrom = 0
avgvolumeto = 0
avgopen = 0
for x in range (1,30):
    for y in prevdays:
        # print(test_data.iloc[((len(test_data)-1)+x)-y])
        high_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['high']
        # print(high_ofprevMonth)
        avghigh = (avghigh + high_ofprevMonth)
        low_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['low']
        avglow = (avglow + low_ofprevMonth)
        vf_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumefrom']
        avgvolumefrom = (avgvolumefrom + vf_ofprevMonth)
        vt_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumeto']
        avgvolumeto = (avgvolumeto + vt_ofprevMonth)
        open_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['open']
        avgopen = (avgopen + open_ofprevMonth)
    avghigh = avghigh/len(prevdays)
    # print(avghigh)
    avglow = avglow/len(prevdays)
    avgvolumefrom = avgvolumefrom/len(prevdays)
    avgvolumeto = avgvolumeto/len(prevdays)
    avgopen = avgopen/len(prevdays)
    test_data = test_data.append({'high' : avghigh, 'low' : avglow, 'volumefrom' : avgvolumefrom, 'volumeto' : avgvolumeto, 'open' : avgopen},ignore_index = True)

saved_final_prediction = load_model('Final prediction modelXRP')

unseen_predictions = predict_model(saved_final_prediction, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
unseen_predictions = unseen_predictions.replace(np.nan, 0)
print(unseen_predictions.tail(40))
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_price'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_label'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# print(crypto)

index = 0
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=ADA&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)
train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)
test_data.drop(["Future_price"], axis='columns', inplace=True)

prevdays =[31,62,93]
avghigh = 0
avglow = 0
avgvolumefrom = 0
avgvolumeto = 0
avgopen = 0
for x in range (1,30):
    for y in prevdays:
        # print(test_data.iloc[((len(test_data)-1)+x)-y])
        high_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['high']
        # print(high_ofprevMonth)
        avghigh = (avghigh + high_ofprevMonth)
        low_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['low']
        avglow = (avglow + low_ofprevMonth)
        vf_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumefrom']
        avgvolumefrom = (avgvolumefrom + vf_ofprevMonth)
        vt_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumeto']
        avgvolumeto = (avgvolumeto + vt_ofprevMonth)
        open_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['open']
        avgopen = (avgopen + open_ofprevMonth)
    avghigh = avghigh/len(prevdays)
    # print(avghigh)
    avglow = avglow/len(prevdays)
    avgvolumefrom = avgvolumefrom/len(prevdays)
    avgvolumeto = avgvolumeto/len(prevdays)
    avgopen = avgopen/len(prevdays)
    test_data = test_data.append({'high' : avghigh, 'low' : avglow, 'volumefrom' : avgvolumefrom, 'volumeto' : avgvolumeto, 'open' : avgopen},ignore_index = True)

saved_final_prediction = load_model('Final prediction modelADA')

unseen_predictions = predict_model(saved_final_prediction, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
unseen_predictions = unseen_predictions.replace(np.nan, 0)
print(unseen_predictions.tail(40))
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_price'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_label'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# print(crypto)

index = 0
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=DOGE&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)
train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)
test_data.drop(["Future_price"], axis='columns', inplace=True)

prevdays =[31,62,93]
avghigh = 0
avglow = 0
avgvolumefrom = 0
avgvolumeto = 0
avgopen = 0
for x in range (1,30):
    for y in prevdays:
        # print(test_data.iloc[((len(test_data)-1)+x)-y])
        high_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['high']
        # print(high_ofprevMonth)
        avghigh = (avghigh + high_ofprevMonth)
        low_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['low']
        avglow = (avglow + low_ofprevMonth)
        vf_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumefrom']
        avgvolumefrom = (avgvolumefrom + vf_ofprevMonth)
        vt_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['volumeto']
        avgvolumeto = (avgvolumeto + vt_ofprevMonth)
        open_ofprevMonth = test_data.iloc[((len(test_data)-1)+x)-y]['open']
        avgopen = (avgopen + open_ofprevMonth)
    avghigh = avghigh/len(prevdays)
    # print(avghigh)
    avglow = avglow/len(prevdays)
    avgvolumefrom = avgvolumefrom/len(prevdays)
    avgvolumeto = avgvolumeto/len(prevdays)
    avgopen = avgopen/len(prevdays)
    test_data = test_data.append({'high' : avghigh, 'low' : avglow, 'volumefrom' : avgvolumefrom, 'volumeto' : avgvolumeto, 'open' : avgopen},ignore_index = True)

saved_final_prediction = load_model('Final prediction modelDOGE')

unseen_predictions = predict_model(saved_final_prediction, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
unseen_predictions = unseen_predictions.replace(np.nan, 0)
print(unseen_predictions.tail(40))
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_price'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# line_plot(unseen_predictions['open'].iloc[0:73], unseen_predictions['prediction_label'].iloc[1:],'actual','prediction', lw=3, xlabel='price[USD]')
# plt.show()
# print(crypto)