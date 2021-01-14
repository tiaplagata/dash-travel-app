# Import required libraries
import os
from random import randint

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd

# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

app = dash.Dash(__name__)
server = app.server

# Put your Dash code here

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


# The app layout
app.layout = html.Div(children=[
    html.H1(children='Where should I travel?'),

    html.Div(children='When traveling becomes a normal passtime again, where should you go? What do you want to do while on vacation?'),

    dcc.Input(id='input_bar',value='ex. I want to go snorkeling and do yoga on the beach',
              placeholder='MSFT,AAPL',
              style={'width': '90%', 'height': 50})

])

# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
#     app.run_server(debug=True)
