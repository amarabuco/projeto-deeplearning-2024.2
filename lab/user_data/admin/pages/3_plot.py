import streamlit as st
import subprocess
import pandas as pd
import os
import mplfinance as mpf
import plotly.graph_objects as go

st.set_page_config(
    layout="wide",
)

st.title('Plotter')


trading_mode = st.radio('trading-mode', ['spot', 'futures'])

if trading_mode == 'spot':
    command = f"freqtrade list-pairs --userdir /freqtrade/user_data/ --exchange binance --all --quote USDT -1"
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    coins = sorted(result.stdout.split('\n'))
else:
    command = f"freqtrade list-pairs --trading-mode futures -1"
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    coins = sorted(result.stdout.split('\n'))
strategies = models = sorted(os.listdir(
    f'/freqtrade/user_data/strategies'))
configs = sorted(os.listdir(f'/freqtrade/user_data/config'))

with st.form(key='download'):
    coin = st.selectbox('Select coins', coins)
    strategy = st.selectbox('Select strategies', strategies)
    config = st.selectbox('configs', configs)
    if st.form_submit_button('Plot'):
        command = f"freqtrade plot-dataframe -p {coin} --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config} --strategy {strategy[:-3]}"
        st.code(command)
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        st.code(result.stdout)
        st.stop()
