from tokenize import Double
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

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root@localhost/crypto"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)
ma = Marshmallow(app)

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BNB&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
# print(hist)

prediction_day = 1

hist['Future_price'] = hist[['close']]
hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
print(hists)

df = hists.copy()
X = np.array(df[df.columns])
X = X[:len(hists)-prediction_day]
y = np.array(df['Future_price'])
y = y[:-prediction_day]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 0, shuffle = False)

train_data = pd.DataFrame(x_train, columns = df.columns)

test_data = pd.DataFrame(x_test, columns = df.columns)

regression_setup = setup(data=train_data, target= 'Future_price', session_id = 123, use_gpu = True)
best_model = compare_models(sort='r2')
model = create_model(best_model)
# evaluate_model(model)

unseen_predictions = predict_model(model, data = test_data)
unseen_predictions = unseen_predictions.append({'high' : np.nan, 'low' : np.nan, 'volumefrom' : np.nan, 'Future_price' : np.nan, 'prediction_label' : np.nan},ignore_index = True)
unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
print('unseen prediction are')
print(unseen_predictions['prediction_price'])

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
line_plot(unseen_predictions['Future_price'], unseen_predictions['prediction_price'],'actual','prediction', lw=3, xlabel='price[USD]')
line_plot(unseen_predictions['Future_price'], unseen_predictions['prediction_label'],'actual','prediction', lw=3, xlabel='price[USD]')
plt.show()

# databse models
class btc(db.Model):
    __tablename__ = 'btc'
    prediction_price = Column(Float)
    price = Column(Float)
class eth(db.Model):
    __tablename__ = 'eth'
    prediction_price = Column(Float)
    price = Column(Float)
class ada(db.Model):
    __tablename__ = 'ada'
    prediction_price = Column(Float)
    price = Column(Float)
class doge(db.Model):
    __tablename__ = 'doge'
    prediction_price = Column(Float)
    price = Column(Float)
class ltc(db.Model):
    __tablename__ = 'ltc'
    prediction_price = Column(Float)
    price = Column(Float)


if __name__ == '__main__':
    app.run()