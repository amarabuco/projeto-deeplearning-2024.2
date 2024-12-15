import os
import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime

st.title('Backtester')

with st.spinner('Loading data...'):
    command = f"freqtrade list-strategies --userdir /freqtrade/user_data/ -1"
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    strategies = sorted(result.stdout.split('\n'))

    configs = sorted(os.listdir(f'/freqtrade/user_data/config'))
    models = sorted(os.listdir(
        f'/freqtrade/user_data/freqaimodels/prediction_models'))

with st.form(key='backtest'):
    freqai = st.radio('freqai', [1, 0])
    trb = st.date_input('inicio', format="YYYY-MM-DD")
    tre = st.date_input('fim', format="YYYY-MM-DD")
    strategy = st.multiselect('strategies', strategies)
    config = st.selectbox('configs', configs)
    freqaimodel = st.multiselect('models', models)
    breakdown = st.multiselect('breakdown', ['day', 'week', 'month'])

    tab1, tab2 = st.columns(2)
    with tab1:
        show = st.form_submit_button('show')
    with tab2:
        run = st.form_submit_button('run', type='primary')

    if show:
        with st.spinner('Showing config...'):
            if freqai == 1:
                command = f"freqtrade backtesting  --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config} --strategy {' '.join(strategy)} --freqaimodel {' '.join(freqaimodel)} --timerange {str(trb).replace('-','')}-{str(tre).replace('-','')} --export trades --export signals --breakdown {' '.join(breakdown)} --cache none"
            else:
                command = f"freqtrade backtesting  --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config} --strategy {' '.join(strategy)} --timerange {str(trb).replace('-','')}-{str(tre).replace('-','')} --export trades --export signals --breakdown {' '.join(breakdown)} --cache none"
            with st.expander('comando'):
                st.code(command)
            result = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            st.code(result.stdout)
        # for pair in pairs:
        #     for time in tf:
        #         if trading_mode == 'spot':
        #             st.write(pd.read_feather(
        #                 f'/freqtrade/user_data/data/binance/{pair.replace('/','_')}-{time}.feather').head())
        #         else:
        #             st.write(pd.read_feather(
        #                 f'/freqtrade/user_data/data/binance/futures/{pair.replace('/','_').replace(":",'_')}-{time}-futures.feather').head())
        st.stop()

    if run:
        with st.spinner('Create backtest...'):
            if freqai == 1:
                command = f"freqtrade backtesting  --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config} --strategy {' '.join(strategy)} --freqaimodel {' '.join(freqaimodel)} --timerange {str(trb).replace('-','')}-{str(tre).replace('-','')} --export trades --export signals --breakdown {' '.join(breakdown)} --cache none"
            else:
                command = f"freqtrade backtesting  --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config} --strategy {' '.join(strategy)} --timerange {str(trb).replace('-','')}-{str(tre).replace('-','')} --export trades --export signals --breakdown {' '.join(breakdown)} --cache none"
            with st.expander('comando'):
                st.code(command)
            # result = subprocess.run(
            #     command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # st.write(result.stdout)
        # for pair in pairs:
        #     for time in tf:
        #         if trading_mode == 'spot':
        #             st.write(pd.read_feather(
        #                 f'/freqtrade/user_data/data/binance/{pair.replace('/','_')}-{time}.feather').head())
        #         else:
        #             st.write(pd.read_feather(
        #                 f'/freqtrade/user_data/data/binance/futures/{pair.replace('/','_').replace(":",'_')}-{time}-futures.feather').head())
        st.stop()
