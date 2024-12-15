import streamlit as st
import subprocess
import pandas as pd

st.title('Downloader')

with st.spinner('Loading data...'):
    trading_mode = st.radio('trading-mode', ['spot', 'futures'])

    command = f"freqtrade list-pairs --userdir /freqtrade/user_data/ --exchange binance --quote USDT  --trading-mode {trading_mode} -1"
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    pares = result.stdout.split('\n')

with st.form(key='download'):
    source = st.radio('source', ['binance', 'yahoo'])
    tf = st.multiselect('timeframes', ["1s", "1m", "3m", "5m", "15m",
                        "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"])
    trb = st.date_input('inicio', format="YYYY-MM-DD")
    tre = st.date_input('fim', format="YYYY-MM-DD")
    pairs = st.multiselect('pairs', pares)
    acao = st.text_input('acao', 'AAPL')

    if st.form_submit_button('Download'):
        if source == 'binance':
            with st.spinner('Downloading data...'):
                command = f"freqtrade download-data --userdir /freqtrade/user_data/ --trading-mode {trading_mode} --exchange binance -p {' '.join(pairs)} --timerange {str(trb).replace('-','')}-{str(tre).replace('-','')} -t {' '.join(tf)}"
                with st.expander('comando'):
                    st.code(command)
                result = subprocess.run(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                st.write(result.stdout)
            for pair in pairs:
                for time in tf:
                    if trading_mode == 'spot':
                        st.write(pd.read_feather(
                            f'/freqtrade/user_data/data/binance/{pair.replace('/','_')}-{time}.feather').head())
                    else:
                        st.write(pd.read_feather(
                            f'/freqtrade/user_data/data/binance/futures/{pair.replace('/','_').replace(":",'_')}-{time}-futures.feather').head())
            st.stop()
        else:
            with st.spinner('Downloading data...'):
                import yfinance as yf
                ticker = yf.Ticker(acao)
                hist = ticker.history(
                    start=str(trb), end=str(tre), interval=tf[0]
                ).reset_index().rename({'Datetime': 'date', 'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, axis=1)
                hist[['date', 'open', 'high', 'low', 'close', 'volume']].to_csv(
                    f'/freqtrade/user_data/data/yahoo/{acao}-{tf[0]}.csv', index=False)
                hist[['date', 'open', 'high', 'low', 'close', 'volume']].to_feather(
                    f'/freqtrade/user_data/data/yahoo/{acao}-{tf[0]}.feather')
                st.code(
                    'Renomei o arquivo com algum par de cripto e mova para a pasta da binance para poder usar os dados. Ex: yahoo/NVDA-1d.feather -> binance/DOGE_USDT-1d.feather')
                st.write(hist)
