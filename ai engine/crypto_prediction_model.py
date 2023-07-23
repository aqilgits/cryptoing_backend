import requests
import json
import pandas as pd
from pycaret.regression import *
import numpy as np
from sklearn.model_selection import train_test_split

cryptos = ["BTC", "ETH", "ADA", "DOGE", "XRP"]
for crypto in cryptos:
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
    model = create_model('ridge')
    save_model(model,'Final prediction model'+crypto)
    # plot_model(model, plot = 'error')
    