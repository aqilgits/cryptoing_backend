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

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:loqil1509.@localhost/crypto"
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
class news(db.Model):
    __tablename__ = 'news'
    num = Column(Integer, primary_key=True)
    sentiment = Column(String)
    title = Column(String)
    url = Column(String)
    
class Price(ma.Schema):
    class Meta:
        fields = ('num','prediction_price', 'price')

price_scheme = Price()
prices_scheme = Price(many=True)

class News(ma.Schema):
    class Meta:
        fields = ('num','sentiment', 'title', 'url')

new_scheme = News()
news_scheme = News(many=True)

# crypto price prediction
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

    saved_final_prediction = load_model('Final prediction model'+crypto)

    unseen_predictions = predict_model(saved_final_prediction, data = test_data)
    unseen_predictions = unseen_predictions.append({'high' : 0, 'low' : 0, 'volumefrom' : 0, 'Future_price' : 0, 'prediction_label' : 0},ignore_index = True)
    unseen_predictions['prediction_price']=unseen_predictions[['prediction_label']].shift(prediction_day)
    unseen_predictions = unseen_predictions.replace(np.nan, 0)
    # print(crypto)
    # print(unseen_predictions)
    
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

# sentiment analysis
url = "https://crypto-news11.p.rapidapi.com/cryptonews/altcoin"

querystring = {"max_articles":"20","last_n_hours":"48","top_n_keywords":"10"}

headers = {
	"X-RapidAPI-Key": "43ca21b1e1msh81aa2e6f1df42a9p19d28ejsnd155bc5d6652",
	"X-RapidAPI-Host": "crypto-news11.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

info = response.text
response_info = json.loads(info)

articles = []
for article_info in response_info['articles']:
    articles.append([article_info['date'],  article_info['sentiment']['polarity'], article_info['sentiment']['subjectivity'], article_info['source'], article_info['subject'],  article_info['text'], article_info['title'], article_info['url']])

article_df = pd.DataFrame(data=articles, columns=['date', 'polarity', 'subjectivity', 'source', 'subject', 'text', 'title', 'url'])
# print(article_df)

saved_final_sentiment = load_model('Final sentiment model')

unseen_predictions_sentiment = predict_model(saved_final_sentiment, data=article_df)
# print(unseen_predictions_sentiment)
count = 0
for x in unseen_predictions_sentiment['prediction_label']:
    num = count
    index_data = news.query.filter_by(num=num).first()
    if index_data:
        index_data.sentiment = x
        index_data.title = article_df['title'][count]
        index_data.url = article_df['url'][count]
    db.session.commit()
    count = count+1

@app.route('/crypto/<string:crypto_name>',methods=['GET'])
def crypto_data(crypto_name:str):
    if crypto_name == 'BTC':
        crypto_data = db.session.execute(db.select(btc).order_by(btc.num)).scalars()
        result = prices_scheme.dump(crypto_data)
    
    elif crypto_name == 'ETH':
        crypto_data = db.session.execute(db.select(eth).order_by(eth.num)).scalars()
        result = prices_scheme.dump(crypto_data)

    elif crypto_name == 'ADA':
        crypto_data = db.session.execute(db.select(ada).order_by(ada.num)).scalars()
        result = prices_scheme.dump(crypto_data)

    elif crypto_name == 'DOGE':
        crypto_data = db.session.execute(db.select(doge).order_by(doge.num)).scalars()
        result = prices_scheme.dump(crypto_data)

    else:
        crypto_data = db.session.execute(db.select(xrp).order_by(xrp.num)).scalars()
        result = prices_scheme.dump(crypto_data)
    
    return jsonify(result)


@app.route('/news',methods=['GET'])
def crypto_news():
    crypto_news = db.session.execute(db.select(news).order_by(news.num)).scalars()
    result = news_scheme.dump(crypto_news)

    return jsonify(result)

@app.route('/news/percentage',methods=['GET'])
def crypto_percentage(): 
    crypto_news = db.session.execute(db.select(news).order_by(news.num)).scalars()
    result = news_scheme.dump(crypto_news)
    sentiment ={}
    sentimentlist =[]
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    for x in result:
        if result[x['num']]['sentiment'] == 'positive':
            count_positive = count_positive +1
        if result[x['num']]['sentiment'] == 'negative':
            count_negative = count_negative +1
        if result[x['num']]['sentiment'] == 'neutral':
            count_neutral = count_neutral +1
    count_positive = (count_positive / 20)*100
    count_negative = (count_negative / 20)*100
    count_neutral = (count_neutral / 20)*100
    sentiment['positive']=round(count_positive)
    sentiment['negative']=round(count_negative)
    sentiment['neutral']=round(count_neutral)
    sentimentlist.append(sentiment)
    print(sentiment)
    return jsonify(sentimentlist)

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(debug=True,host='localhost', port=port)