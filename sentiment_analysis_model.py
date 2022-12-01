from pycaret.classification import *
import pandas as pd
import json
import requests

# sentiment analysis
url = "https://crypto-news11.p.rapidapi.com/cryptonews/bitcoin"

querystring = {"max_articles":"10","last_n_hours":"48","top_n_keywords":"10"}

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
print(article_df)
# news dataset
dataset = pd.read_csv("cryptonews.csv")
dataset = dataset.head(1000)

dataset.drop(["date","source"], axis='columns', inplace=True)
dataset.dropna()
data = dataset.sample(frac=0.80, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

clasification_setup = setup(data = data, target = 'sentiment', session_id=123)

best_model = compare_models()

model = create_model(best_model)

tuned_model = tune_model(best_model)

final_model = finalize_model(tuned_model)
save_model(final_model,'Final sentiment model')

