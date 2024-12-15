import os
import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime

st.set_page_config(page_title=None, page_icon=None, layout="wide",
                   initial_sidebar_state="auto", menu_items=None)

st.title('Hyperopter')

with st.spinner('Loading data...'):

    hyperopt_dir = f'/freqtrade/user_data/hyperopt_results/'
    files = sorted([f for f in os.listdir(hyperopt_dir)])

    configs = sorted(os.listdir(f'/freqtrade/user_data/config'))
    models = sorted(os.listdir(
        f'/freqtrade/user_data/freqaimodels/prediction_models'))

    filename = st.selectbox('strategies', files)

    if st.button('show'):

        with st.spinner('Carregando...'):
            command = f"freqtrade hyperopt-list --userdir /freqtrade/user_data/ --hyperopt-filename {filename} --export-csv /freqtrade/user_data/admin/hyperopt.csv"
            result = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with st.expander('comando'):
                st.code(command)
            with st.expander('épocas'):
                df = pd.read_csv('/freqtrade/user_data/admin/hyperopt.csv')
                st.write(df)
            n = df.sort_values('Objective', ascending=False)['Epoch'].values[0]

            command = f"freqtrade hyperopt-show --userdir /freqtrade/user_data/ --hyperopt-filename {filename} --best -n -1 --no-header"
            with st.expander('comando'):
                st.code(command)
            with st.expander(f'Época'):
                try:
                    result = subprocess.run(
                        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    st.text(
                        open("/freqtrade/user_data/admin/hyperopt.txt").read())
                except:
                    st.write("erro")
