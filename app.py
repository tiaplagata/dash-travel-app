# Import required libraries
import os
from random import randint

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd
import numpy as np

from jupyter_dash import JupyterDash
import ipywidgets

from joblib import load

import pickle
import regex as re
import string
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

app = dash.Dash(__name__)
server = app.server

# Dash code here

data = pd.read_csv('./assets/cities_df', index_col=0)
X = data['Attraction']
y = data['City']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

def preprocess_df(df, column, preview=True):
    """
    Input df with raw text attractions.
    Return df with preprocessed text.
    """
    
    df[column] = df['Attraction'].apply(lambda x: x.lower())
    df[column] = df[column].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    df[column] = df[column].apply(lambda x: re.sub('\w*\d\w*','', x))
    
    return df

X_train_cleaned = preprocess_df(pd.DataFrame(X_train, columns=['Attraction']),
                                'cleaned')


new_stopwords = stopwords.words('english') + list(string.punctuation)
new_stopwords += ['bali', 'barcelona', 'crete', 'dubai', 'istanbul', 'london',
                  'majorca', 'phuket', 'paris', 'rome', 'sicily', 'mallorca',
                  'goa', 'private', 'airport', 'transfer']

vectorizer = TfidfVectorizer(analyzer='word',
                             stop_words=new_stopwords,
                             decode_error='ignore')
                                
X_train_tfidf = vectorizer.fit_transform(X_train_cleaned['cleaned'])
    
model = load('./assets/non_lemmatized_model')

def preprocess_text(text):
    """
    Input raw text.
    Return preprocessed text.
    """
    
    preprocessed = text.lower()
    preprocessed = re.sub('[%s]' % re.escape(string.punctuation), '', preprocessed)
    preprocessed = re.sub('\w*\d\w*','', preprocessed)
        
    return [preprocessed]

def get_prediction(raw_text):
    try:
        preprocessed_text = preprocess_text(raw_text)
        probas = model.predict_proba(vectorizer.transform(preprocessed_text))
        classes = model.classes_
        first_pred = classes[probas.argmax()]
        second_pred = classes[np.argsort(probas)[:, 10]][0]
        return first_pred, second_pred
    except:
        pass
    

bali_wordcloud = './assets/bali_wordcloud.png'
barcelona_wordcloud = './assets/barcelona_wordcloud.png'
crete_wordcloud = './assets/crete_wordcloud.png'
dubai_wordcloud = './assets/dubai_wordcloud.png'
goa_wordcloud = './assets/goa_wordcloud.png'
istan_wordcloud = './assets/istanbul_wordcloud.png'
london_wordcloud = './assets/london_wordcloud.png'
majorca_wordcloud = './assets/majorca_wordcloud.png'
paris_wordcloud = './assets/paris_wordcloud.png'
phuket_wordcloud = './assets/phuket_wordcloud.png'
rome_wordcloud = './assets/rome_wordcloud.png'
sicily_wordcloud = './assets/sicily_wordcloud.png'
    
# The app layout
app.layout = html.Div(children=[
    html.H1(children='Where should I travel?'),

    html.Div(children='When traveling becomes a normal passtime again, where should you go?'),
    
    html.Br(),
    
    html.Div(["What do you want to do on vacation? ",
              dcc.Input(id='my-input', style={'width':'90%'}, placeholder= 'ex. I want to go to the beach',
                        value='', type='text')]),
    html.Hr(),
    
    html.Div(id='my-output'),
    
    html.Br(),
    
    html.Img(id='image')


    
])
    

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'))
def update_output_div(input_value):
    first_pred, second_pred = get_prediction(input_value)        
    return first_pred


@app.callback(
    Output(component_id='image', component_property='src'),
    Input('my-output', 'children'))
def update_image(city):
    if city == 'Rome, Italy':
        city_img = rome_wordcloud
    elif city == 'Crete, Greece':
        city_img = crete_wordcloud
    elif city == 'Paris, France':
        city_img = paris_wordcloud
    elif city == 'Bali, Indonesia':
        city_img = bali_wordcloud
    elif city == 'Majorca, Balearic Islands':
        city_img = majorca_wordcloud
    elif city == 'Phuket, Thailand':
        city_img = phuket_wordcloud
    elif city == 'Barcelona, Spain':
        city_img = barcelona_wordcloud
    elif city == 'Dubai, United Arab Emirates':
        city_img = dubai_wordcloud
    elif city == 'Sicily, Italy':
        city_img = sicily_wordcloud
    elif city == 'Goa, India':
        city_img = goa_wordcloud
    elif city == 'Istanbul, Turkey':
        city_img = istan_wordcloud
    else:
        city_img = london_wordcloud
    return city_img

# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
#     app.run_server(debug=True)
