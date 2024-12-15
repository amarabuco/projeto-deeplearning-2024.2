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

from chronos import ChronosPipeline
import torch


logger = logging.getLogger(__name__)


class ChronosRLStrategy(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.

    Launching this strategy would be:

    freqtrade trade --strategy FreqaiExampleHybridStrategy --strategy-path freqtrade/templates
    --freqaimodel CatboostClassifier --config config_examples/config_freqai.example.json

    or the user simply adds this to their config:

    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 15,
        "identifier": "unique-id",
        "feature_parameters": {
            "include_timeframes": [
                "3m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    Thanks to @smarmau and @johanvulgt for developing and sharing the strategy.
    """

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

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        def get_embeddings(data):
            # Convert all columns of the dataframe to numeric types if possible.
            # If not, errors='coerce' will replace non-numeric values with NaN.
            # Explicitly set the dtype to a supported type
            context = torch.from_numpy(data.astype(np.float32))
            embeddings, tokenizer_state = pipeline.embed(context)
            return embeddings.type(torch.float32).reshape(-1).numpy()

        rows = []
        for row in dataframe[["%-pct-close", "%-pct-volume"]].iterrows():
            tmp = list(get_embeddings(row[1].values))
            rows.append(tmp)
        new_data = pd.DataFrame(rows)

        new_data.columns = ['%-embeddings_' +
                            str(col) for col in new_data.columns]

        dataframe = pd.concat([dataframe, new_data], axis=1)

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
