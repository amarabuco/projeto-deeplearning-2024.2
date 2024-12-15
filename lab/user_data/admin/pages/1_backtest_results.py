import os
import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime
import plotly.express as px
from pathlib import Path

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.plot.plotting import generate_candlestick_graph

import talib.abstract as ta
from technical import qtpylib

st.set_page_config(page_title=None, page_icon=None, layout="wide",
                   initial_sidebar_state="auto", menu_items=None)

st.title('Backtester Results')

with st.spinner('Loading data...'):
    backtest_dir = f'/freqtrade/user_data/backtest_results/'
    files = sorted([f for f in os.listdir(backtest_dir)
                   if f[-4:] == 'json' and not "meta" in f and not "trades" in f], reverse=True)

    compare = st.radio('compare', [0, 1])
    # trb = st.date_input('inicio', format="YYYY-MM-DD")
    # tre = st.date_input('fim', format="YYYY-MM-DD")
    filename = st.selectbox('strategies', files)
    stats = load_backtest_stats(backtest_dir + filename)
    strategies = stats["strategy"].keys()
    strategy = st.selectbox('strategies', strategies)
    # config = st.selectbox('configs', configs)
    # freqaimodel = st.multiselect('models', models)
    # # breakdown = st.multiselect('models', ['day', 'week', 'month'])
    # signal = st.selectbox('signals', signals)
    # m = st.selectbox('m', market)
    # trade = st.selectbox('trade', trades)

    show = st.button('show')

if show:

    if len(strategy) == 1:
        strategy = strategy[0]
    # st.write(strategy)

    # st.write(stats)
    with st.expander("Raw"):
        st.json(stats)
    with st.expander("metrics"):
        st.write("total_trades:", '{:.0f}'.format(
            stats["strategy"][strategy]["total_trades"]))
        st.write("wins:", '{:.0f}'.format(stats["strategy"][strategy]["wins"]))
        st.write("draws:", '{:.0f}'.format(
            stats["strategy"][strategy]["draws"]))
        st.write("losses:", '{:.0f}'.format(
            stats["strategy"][strategy]["losses"]))
        st.write("max_consecutive_wins:",
                 '{:.0f}'.format(stats["strategy"][strategy]
                                 ["max_consecutive_wins"]))
        st.write("max_consecutive_losses:",
                 '{:.0f}'.format(stats["strategy"][strategy]
                                 ["max_consecutive_losses"]))
        st.write("winrate:", '{:.2%}'.format(
            stats["strategy"][strategy]["winrate"]))
        st.write("profit_total_abs:", '{:,.2f}'.format(
            stats["strategy"][strategy]["profit_total_abs"]))
        st.write("profit_total:", '{:.2%}'.format(
            stats["strategy"][strategy]["profit_total"]))
        st.write("profit_mean:", '{:.2%}'.format(
            stats["strategy"][strategy]["profit_mean"]))
        st.write("profit_median:", '{:.2%}'.format(
            stats["strategy"][strategy]["profit_median"]))
        st.write("profit_factor:",
                 '{:.2f}'.format(stats["strategy"][strategy]["profit_factor"]))
        st.write("max_drawdown_account:", '{:.2%}'.format(
            stats["strategy"][strategy]["max_drawdown_account"]))
        st.write("max_drawdown_abs:", '{:,.2f}'.format(
            stats["strategy"][strategy]["max_drawdown_abs"]))
        st.write("total_volume:", '{:,.0f}'.format(
            stats["strategy"][strategy]["total_volume"]))
        st.write("cagr:", '{:.2%}'.format(stats["strategy"][strategy]["cagr"]))
        st.write("sortino:", '{:.2f}'.format(
            stats["strategy"][strategy]["sortino"]))
        st.write("sharpe:", '{: .2f}'.format(
            stats["strategy"][strategy]["sharpe"]))
        st.write("calmar:", '{:.2f}'.format(
            stats["strategy"][strategy]["calmar"]))
        st.write("profit_factor:", '{: .2f}'.format(
                 stats["strategy"][strategy]["profit_factor"]))

    with st.expander("Pairs"):
        st.dataframe(stats["strategy"][strategy]["results_per_pair"])
        # Get pairlist used for this backtest
        st.dataframe(stats["strategy"][strategy]["pairlist"])
    with st.expander("market_change"):
        # Get market change (average change of all pairs from start to end of the backtest period)
        st.metric(label="Market Change",
                  value='{:.2%}'.format(stats["strategy"][strategy]["market_change"]))
    with st.expander("drawdown"):
        # Maximum drawdown ()
        st.write(stats["strategy"][strategy]["max_relative_drawdown"])
        st.write(stats["strategy"][strategy]["max_drawdown_abs"])
        # # Maximum drawdown start and end
        st.write(stats["strategy"][strategy]["drawdown_start"])
        st.write(stats["strategy"][strategy]["drawdown_end"])
    with st.expander("trades"):
        # Load backtested trades as dataframe
        trades = load_backtest_data(backtest_dir + filename)
        st.dataframe(trades, use_container_width=True)
    with st.expander("breakdown"):
        # Get periodic breakdown
        tab1, tab2, tab3 = st.tabs(["day", "week", "month"])

        with tab1:
            if "day" in stats["strategy"][strategy]["periodic_breakdown"].keys():
                st.dataframe(stats["strategy"][strategy]
                             ["periodic_breakdown"]["day"], use_container_width=True)
                fig_day = px.line(stats["strategy"][strategy]
                                  ["periodic_breakdown"]["day"], x="date", y="profit_abs", markers=True)
                st.plotly_chart(fig_day, use_container_width=True, key='day')

        with tab2:
            if "week" in stats["strategy"][strategy]["periodic_breakdown"].keys():
                st.dataframe(stats["strategy"][strategy]
                             ["periodic_breakdown"]["week"],  use_container_width=True)
                fig_week = px.bar(stats["strategy"][strategy]
                                  ["periodic_breakdown"]["week"], x="date", y="profit_abs")
                st.plotly_chart(
                    fig_week, use_container_width=True, key='week')
        with tab3:
            if "month" in stats["strategy"][strategy]["periodic_breakdown"].keys():
                st.write("month")
                st.dataframe(stats["strategy"][strategy]
                             ["periodic_breakdown"]["month"],  use_container_width=True)
                fig_month = px.bar(stats["strategy"][strategy]
                                   ["periodic_breakdown"]["month"], x="date", y="profit_abs")
                st.plotly_chart(
                    fig_month, use_container_width=True, key='month')

    with st.expander("trades pair"):
        # Show value-counts per pair
        # st.write(trades.groupby("pair")["enter_reason"].value_counts())
        st.write(stats["strategy"][strategy]["results_per_pair"][0][
            "profit_total_abs"])
        st.write(trades.groupby("pair")[
            "enter_tag"].value_counts())
        st.write(trades.groupby("pair")["exit_reason"].value_counts())
        # st.write(stats["strategy"][strategy]["mix_tag_stats"].groupby(
        #     "key")["trades"])

    with st.expander("Trades Chart"):
        trades_red = trades.loc[trades["pair"] ==
                                stats["strategy"][strategy]["pairlist"][0]]

        tf = stats["strategy"][strategy]["timeframe"]

        pair = stats["strategy"][strategy]["pairlist"][0]

        data = load_pair_history(
            datadir=Path(f'/freqtrade/user_data/data/binance'),
            timeframe=tf,
            pair=pair,
            data_format="feather",  # Make sure to update this to your data
            candle_type=CandleType.SPOT,
        )

        data = data.set_index("date", drop=False)

        data_red = data[stats["strategy"][strategy]["backtest_start"]
            :stats["strategy"][strategy]["backtest_end"]]

        # st.write(data_red)
        # st.write(trades_red)

        # data_red = data_red[data_red.volume >
        #                     0]

        # trades_red = trades_red[trades_red['close_date'].isin(data_red.date)]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(data_red), window=20, stds=2)
        data_red["bb_lowerband"] = bollinger["lower"]
        data_red["bb_middleband"] = bollinger["mid"]

        data_red["bb_upperband"] = bollinger["upper"]
        data_red["bb_percent"] = (data_red["close"] - data_red["bb_lowerband"]) / (
            data_red["bb_upperband"] - data_red["bb_lowerband"]
        )
        data_red["bb_width"] = (data_red["bb_upperband"] - data_red["bb_lowerband"]) / data_red[
            "bb_middleband"
        ]
        data_red["%-sma-period"] = ta.SMA(data_red, timeperiod=10)
        data_red["%-ema-period"] = ta.EMA(data_red, timeperiod=20)

        # Generate candlestick graph
        graph = generate_candlestick_graph(
            pair=pair,
            data=data_red,
            trades=trades_red,
            indicators1=["%-sma-period", "%-ema-period"],
        )

        st.plotly_chart(graph)

    try:
        with st.expander("Profit"):
            df = pd.DataFrame(columns=["0", "1"],
                              data=stats["strategy"][strategy]["daily_profit"])
            df.rename({'0': 'dates', '1': 'equity'}, axis=1, inplace=True)
            df["equity_cumsum"] = df["equity"].cumsum()

            fig = px.area(df, x="dates", y="equity_cumsum")
            st.plotly_chart(fig, use_container_width=True)
    except:
        pass

    with st.expander("Histogram"):
        # st.write(trades_red)
        # df = pd.DataFrame(columns=["0", "1"],
        #                   data=trades_red["profit_ratio"])
        # st.write(df)
        # df.rename({'0': 'dates', '1': 'profit'}, axis=1, inplace=True)
        # st.write(df)
        fig = px.histogram(trades, x="profit_ratio")
        st.plotly_chart(fig, use_container_width=True)
