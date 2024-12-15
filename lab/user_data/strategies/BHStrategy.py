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
import mlflow
import copy

logger = logging.getLogger(__name__)


class BHStrategy(IStrategy):

    # ## timeframe
    timeframe = "1d"

    minimal_roi = {
        "0": 10
    }

    process_only_new_candles = True
    # ## Stoploss

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -1
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.017  # Disabled / not configured
    trailing_only_offset_is_reached = False

    # use_custom_stoploss = False

    use_custom_stoploss = False

    # ## Exit

    use_exit_signal = False

    exit_profit_only = False

    exit_profit_offset = 0.0

    ignore_roi_if_entry_signal = True

    # ## Candles

    process_only_new_candles = True
    startup_candle_count: int = 0

    # ## Short

    can_short = False

    unfilledtimeout = {
        "unit": "minutes",
        "entry": 30,
        "exit": 30,
        "exit_timeout_count": 0
    }

    # ## Candles

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        mlflow.set_tracking_uri("/freqtrade/user_data/mlruns")
        # logger.info(Path(dk.data_path / "mlruns"))
        mlflow.set_experiment("PAPER")
        # mlflow.autolog()

        with mlflow.start_run(run_name=f"bh_{metadata['pair']}") as run:
            cfg = copy.deepcopy(self.config)
            cfg.pop('api_server')
            cfg.pop('telegram')
            cfg.pop('original_config')
            mlflow.log_params(cfg)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # enter_long_conditions = [
        #     df["do_predict"] == 1, df["&-action"] == 1]
        enter_long_conditions = [df["close"] > 0]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), [
                    "enter_long"]
            ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        exit_long_conditions = [df["close"] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions),
                   "exit_long"] = 1

        return df
