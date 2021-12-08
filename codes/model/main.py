import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from fastai.text.all import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from gensim import models
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings("ignore")
from keras.initializers import glorot_uniform
import tensorflow as tf

#Reading the model from JSON file
with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('model.h5')

# Use fastai to handle text preprocessing and tokenization
from fastai.text.all import *
path = Path(f'{os.getcwd()}/../datasets/')

tweets_covid_all_vaccination = pd.read_csv(path/'tweets_covid_all_vaccination.csv')
tweets_extraction = pd.read_csv(path/'tweets_extraction.csv')

# Remove Emojis Helper
def remove_emojis(str):
    return str.encode('ascii', 'ignore').decode('ascii')

# Remove URLs, Hashtags, handles, and Emojis
def remove(ts, idx='text'):
    ts['orig_text'] = ts[idx]
    ts[idx] = ts[idx].apply(lambda x:re.sub('@[^\s]+','',x))
    ts[idx] = ts[idx].apply(lambda x:re.sub(r"http\S+", "", x))
    ts[idx] = ts[idx].apply(remove_emojis)
    ts[idx] = ts[idx].apply(lambda x:re.sub(r'\B#\S+','',x))
    return ts[ts[idx]!='']

tweets_covid_all_vaccination['sentiment'] = np.nan
tweets_covid_all_vaccination = remove(tweets_covid_all_vaccination)
tweets_extraction = tweets_extraction[['old_text', 'new_sentiment']].rename(columns={'old_text':'text', 'new_sentiment':'sentiment'})
tweets_extraction = remove(tweets_extraction)
tweets_merge = tweets_extraction[['text', 'sentiment']].append(tweets_covid_all_vaccination[['text', 'sentiment']])
tweets = tweets_merge.dropna(subset=['sentiment'])

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['text_tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

word2vec = models.KeyedVectors.load_word2vec_format('/home/madao/CS410/GoogleNews-vectors-negative300.bin.gz', binary=True)

test_text = tweets_covid_all_vaccination[['text']]
stopwords_list = stopwords.words('english')
all_text_tokens = []
for sentence in test_text["text"]:
    content_token = nltk.word_tokenize(sentence)

    lower_token = []
    for token in content_token:
        lower_token.append(token.lower())

    punctuation_token = []
    for token in lower_token:
        punctuation_token.append(re.sub(r'[^\w\s]+', '', token))

    small_token = []
    for token in punctuation_token:
        if len(token)>1:
            small_token.append(token)

    stop_token = []
    for token in small_token:
        if token not in stopwords_list:
            stop_token.append(token)

    lemmatization_token = []
    for token in stop_token:
        lemmatization_token.append(WordNetLemmatizer().lemmatize(token))

    stemming_token = []
    for token in lemmatization_token:
        stemming_token.append(PorterStemmer().stem(token))
    all_text_tokens.append(stemming_token)
test_text['text_final'] = [' '.join(text) for text in all_text_tokens]
test_text['text_tokens'] = all_text_tokens
test_embeddings = get_word2vec_embeddings(word2vec, test_text, generate_missing=True)
test_words = [word for tokens in test_text["text_tokens"] for word in tokens]
test_vocs = sorted(list(set(test_words)))
from keras.preprocessing.text import Tokenizer
text_final_list = test_text["text_final"].tolist()
tokenizer = Tokenizer(num_words=len(test_vocs), char_level=False)
tokenizer.fit_on_texts(text_final_list)
test_sequences = tokenizer.texts_to_sequences(text_final_list)
test_tokens = tokenizer.word_index
MAX_SEQUENCE_LENGTH = 65
EMBEDDING_DIM = 300
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
result = model_j.predict(test_cnn_data)
tweets_covid_all_vaccination['date'] = pd.to_datetime(tweets_covid_all_vaccination['date'], errors='coerce').dt.date
tweets_covid_all_vaccination['sentiment'] = list(result.argmax(axis=-1))
tweets_covid_all_vaccination['sentiment'] = tweets_covid_all_vaccination['sentiment'].map({0:'negative', 1:'neutral', 2:'positive'})
date_sentiment = tweets_covid_all_vaccination.groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()
data_df = pd.read_csv(f"{os.getcwd()}/../vaccination_datasets/country_vaccinations.csv")
#Data processing referenced from https://www.kaggle.com/gpreda/covid-19-vaccination-progress

country_vaccine_time = data_df[["country", "vaccines", "date", 'total_vaccinations', 
                                'total_vaccinations_per_hundred',  'people_vaccinated','people_vaccinated_per_hundred',
                               'daily_vaccinations','daily_vaccinations_per_million', 
                                'people_fully_vaccinated', 'people_fully_vaccinated_per_hundred'
                               ]].dropna()
country_vaccine_time.columns = ["Country", "Vaccines", "Date", 'Total vaccinations', 'Percent', 'People vaccinated', 'People percent',
                               "Daily vaccinations", "Daily vaccinations per million", 
                                'People fully vaccinated', 'People fully vaccinated percent']
countries = ['Austria', 'Belgium', 'Bulgaria','Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
             'Greece', 'Hungary', 'Ireland', 'Israel', 'Italy', 'Latvia','Lithuania', 'Luxembourg', 'Malta',
             'Netherlands', 'Norway','Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Spain', 'Sweden',
             'United Kingdom', 'United States']
new_country_vaccine = country_vaccine_time[country_vaccine_time['Country']=='United States'][['Date', 'Percent', 'Country']]
import datetime
date_sentiment = date_sentiment[(date_sentiment['date'] > datetime.date(year=2021,month=1,day=20)) & (date_sentiment['date'] < datetime.date(year=2021,month=3,day=16))]
new_country_vaccine['Date'] = pd.to_datetime(new_country_vaccine['Date']).dt.date
new_country_vaccine = new_country_vaccine[(new_country_vaccine['Date'] > datetime.date(year=2021,month=1,day=20)) & (new_country_vaccine['Date'] < datetime.date(year=2021,month=3,day=16))]
new_country_vaccine_daily_percent = country_vaccine_time[country_vaccine_time['Country']=='United States'][['Date', 'Daily vaccinations per million', 'Country']]
new_country_vaccine_daily_percent['Date'] = pd.to_datetime(new_country_vaccine_daily_percent['Date']).dt.date
new_country_vaccine_daily_percent = new_country_vaccine_daily_percent[(new_country_vaccine_daily_percent['Date'] > datetime.date(year=2021,month=1,day=20)) & (new_country_vaccine_daily_percent['Date'] < datetime.date(year=2021,month=3,day=16))]

subfig = make_subplots(specs=[[{"secondary_y": True}]])

fig = px.line(date_sentiment, x='date', y='tweets', color='sentiment', category_orders={'sentiment': ['neutral', 'negative', 'positive']},
             title='Timeline showing sentiment of tweets about COVID-19 vaccines')
fig2 = px.line(new_country_vaccine_daily_percent, x='Date', y=['Daily vaccinations per million'])
fig2.update_traces(yaxis="y2")

subfig.add_traces(fig.data + fig2.data)
subfig.layout.xaxis.title="Time"
subfig.layout.yaxis.title="sentiment"
subfig.layout.yaxis2.title="daily vaccine percent"
subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
subfig.show()
plot(subfig, filename='correlation_percent.html')

new_country_vaccine_daily = country_vaccine_time[country_vaccine_time['Country']=='United States'][['Date', 'Daily vaccinations', 'Country']]
new_country_vaccine_daily['Date'] = pd.to_datetime(new_country_vaccine_daily['Date']).dt.date
new_country_vaccine_daily = new_country_vaccine_daily[(new_country_vaccine_daily['Date'] > datetime.date(year=2021,month=1,day=20)) & (new_country_vaccine_daily['Date'] < datetime.date(year=2021,month=3,day=16))]

subfig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.line(date_sentiment, x='date', y='tweets', color='sentiment', category_orders={'sentiment': ['neutral', 'negative', 'positive']})
fig2 = px.line(new_country_vaccine_daily, x='Date', y=['Daily vaccinations'])

fig2.update_traces(yaxis="y2")

subfig.add_traces(fig.data + fig2.data)
subfig.layout.xaxis.title="Time"
subfig.layout.yaxis.title="sentiment"
subfig.layout.yaxis2.title="daily vaccines"
subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
subfig.show()
plot(subfig, filename='correlation_number.html')