import logging
from typing import Dict
from functools import reduce
from typing import Optional, Union

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from datetime import datetime
from pandas import Series

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair  # noqa
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions

logger = logging.getLogger(__name__)


class DoubleMARLStrategy(IStrategy):

    # ## timeframe
    timeframe = "1d"

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        "0": 0.2,
        "360": 0.1,
        "720": 0
    }

    plot_config = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Up_or_down": {
                "&s-up_or_down": {"color": "green"},
            },
            "&s-up_or_down_short": {
                "&s-up_or_down_short": {"color": "yellow"},
            },
        },
    }

    process_only_new_candles = True
    # ## Stoploss

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.017  # Disabled / not configured
    trailing_only_offset_is_reached = False

    # use_custom_stoploss = False

    use_custom_stoploss = False

    # ## Exit

    use_exit_signal = True

    exit_profit_only = False

    exit_profit_offset = 0.0

    ignore_roi_if_entry_signal = True

    # ## Candles

    process_only_new_candles = True
    startup_candle_count: int = 30

    # ## Short

    can_short = False

    unfilledtimeout = {
        "unit": "minutes",
        "entry": 30,
        "exit": 30,
        "exit_timeout_count": 0
    }

    # ## Candles

    process_only_new_candles = True
    startup_candle_count: int = 30

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30,
                           space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70,
                            space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70,
                             space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(
        low=1, high=50, default=30, space="buy", optimize=True, load=True)
    di_max = IntParameter(low=1, high=20, default=10,
                          space='buy', optimize=True, load=True)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:

        dataframe["%-pct-close"] = dataframe["close"].pct_change()
        dataframe["%-pct-volume"] = dataframe["volume"].pct_change()

        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe["&-action"] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # enter_long_conditions = [
        #     df["do_predict"] == 1, df["&-action"] == 1]
        enter_long_conditions = [df["&-action"] == 1]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), [
                    "enter_long"]
            ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        exit_long_conditions = [df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions),
                   "exit_long"] = 1

        return df
