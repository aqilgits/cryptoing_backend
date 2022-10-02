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

# databse models
class btc(db.Model):
    __tablename__ = 'btc'
    num = Column(Integer, primary_key=True)
    prediction_price = Column(Float)
    price = Column(Float)
class eth(db.Model):
    __tablename__ = 'eth'
    num = Column(Integer, primary_key=True)
    prediction_price = Column(Float)
    price = Column(Float)
class ada(db.Model):
    __tablename__ = 'ada'
    num = Column(Integer, primary_key=True)
    prediction_price = Column(Float)
    price = Column(Float)
class doge(db.Model):
    __tablename__ = 'doge'
    num = Column(Integer, primary_key=True)
    prediction_price = Column(Float)
    price = Column(Float)
class xrp(db.Model):
    __tablename__ = 'xrp'
    num = Column(Integer, primary_key=True)
    prediction_price = Column(Float)
    price = Column(Float)

class Btc(ma.Schema):
    class Meta:
        fields = ('preds', 'price')
class Eth(ma.Schema):
    class Meta:
        fields = ('preds', 'price')
class Ada(ma.Schema):
    class Meta:
        fields = ('preds', 'price')
class Doge(ma.Schema):
    class Meta:
        fields = ('preds', 'price')
class Xrp(ma.Schema):
    class Meta:
        fields = ('preds', 'price')

btc_scheme = Btc()
btc_scheme = Btc(many=True)
eth_scheme = Eth()
eth_scheme = Eth(many=True)
ada_scheme = Ada()
ada_scheme = Ada(many=True)
doge_scheme = Doge()
doge_scheme = Doge(many=True)
xrp_scheme = Xrp()
xrp_scheme = Xrp(many=True)

cryptos = ["BTC", "ETH", "ADA", "DOGE", "XRP"]
for crypto in cryptos:
    index = 0
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym='+crypto+'&tsym=USD&limit=500')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'
    hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
    # print(hist)

    prediction_day = 1

    hist['Future_price'] = hist[['close']]
    hists = hist[['high','low','volumefrom', 'volumeto','open','Future_price']]
    # print(hists)

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
    unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
    unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
    unseen_predictions = unseen_predictions.replace(np.nan, 0)
    print(crypto)
    print(unseen_predictions)

    for x in unseen_predictions['prediction_price']:
        num = index
        prediction_price = x
        price = unseen_predictions['Future_price'][index]
        if crypto == 'BTC':
            latest_price = btc(num=num,
            prediction_price=prediction_price,
            price=price)
        elif crypto == 'ETH':
            latest_price = eth(num=num,
            prediction_price=prediction_price,
            price=price)
        elif crypto == 'ADA':
            latest_price = ada(num=num,
            prediction_price=prediction_price,
            price=price)
        elif crypto == 'DOGE':
            latest_price = doge(num=num,
            prediction_price=prediction_price,
            price=price)
        else:
            latest_price = xrp(num=num,
            prediction_price=prediction_price,
            price=price)

        db.session.add(latest_price)
        db.session.commit()
        index = index+1


# # def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, xlabel=None, ylabel=None):
# #     fig, ax = plt.subplots(1, figsize=(13, 7))
# #     ax.plot(line1, label=label1, linewidth=lw)
# #     ax.plot(line2, label=label2, linewidth=lw)
# #     ax.set_ylabel(xlabel, fontsize=14)
# #     ax.set_title(title, fontsize=16)
# #     ax.legend(loc='best', fontsize=16)
# #     ax.set_ylabel(ylabel, fontsize=14)
# # line_plot(unseen_predictions['Future_price'], unseen_predictions['prediction_price'],'actual','prediction', lw=3, xlabel='price[USD]')
# # line_plot(unseen_predictions['Future_price'], unseen_predictions['prediction_label'],'actual','prediction', lw=3, xlabel='price[USD]')
# # plt.show()


if __name__ == '__main__':
    app.run()