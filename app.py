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
    
    if crypto == 'BTC':
        for x in unseen_predictions['prediction_price']:
            num = index
            index_data = btc.query.filter_by(num=num).first()
            if index_data:
                index_data.prediction_price = x
                index_data.price = unseen_predictions['Future_price'][index]
            db.session.commit()
            index = index+1

    elif crypto == 'ETH':
        for x in unseen_predictions['prediction_price']:
            num = index
            index_data = eth.query.filter_by(num=num).first()
            if index_data:
                index_data.prediction_price = x
                index_data.price = unseen_predictions['Future_price'][index]
            db.session.commit()
            index = index+1
    elif crypto == 'ADA':
        for x in unseen_predictions['prediction_price']:
            num = index
            index_data = ada.query.filter_by(num=num).first()
            if index_data:
                index_data.prediction_price = x
                index_data.price = unseen_predictions['Future_price'][index]
            db.session.commit()
            index = index+1
    elif crypto == 'DOGE':
        for x in unseen_predictions['prediction_price']:
            num = index
            index_data = doge.query.filter_by(num=num).first()
            if index_data:
                index_data.prediction_price = x
                index_data.price = unseen_predictions['Future_price'][index]
            db.session.commit()
            index = index+1
    else:
        for x in unseen_predictions['prediction_price']:
            num = index
            index_data = xrp.query.filter_by(num=num).first()
            if index_data:
                index_data.prediction_price = x
                index_data.price = unseen_predictions['Future_price'][index]
            db.session.commit()
            index = index+1

if __name__ == '__main__':
    app.run()