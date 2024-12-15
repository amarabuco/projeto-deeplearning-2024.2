import os
import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime
import json

st.title('Backtester')

with st.spinner('Loading data...'):
    configs = sorted(os.listdir(f'/freqtrade/user_data/config'))

with st.form(key='backtest'):

    config = st.selectbox('configs', configs)

    tab1, tab2 = st.columns(2)
    with tab1:
        show = st.form_submit_button('show')
    with tab2:
        run = st.form_submit_button('run', type='primary')

    if show:
        command = f"freqtrade test-pairlist --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config}"
        config_content = json.load(
            open(f'/freqtrade/user_data/config/{config}'))
        st.json(config_content['pairlists'])
        st.code(command)
        # result = subprocess.run(
        #     command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # st.code(result.stdout)

    if run:
        command = f"freqtrade test-pairlist --userdir /freqtrade/user_data/ --config /freqtrade/user_data/config/{config}"
        with st.expander('comando'):
            st.code(command)
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        st.code(result.stdout)
