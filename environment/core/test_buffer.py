import rethinkdb as r
import numpy as np
import xarray
import pandas as pd
import json
import time
import asyncio
import constants.db as db_const
import constants.fields as fields

class TestBuffer():
    def __init__(
        self,
        quote_asset="BTC",
        r_host="localhost",
        r_port=28015,
        r_user="admin",
        r_pass="",
        r_db="binance",
        r_feature_table="features",
        r_prices_table="prices",
        r_info_table="info"
    ):
       
        self.feature_table = r_feature_table
        self.prices_table = r_prices_table
        self.quote_asset = quote_asset

        self.index_columns = [
            'baseAsset',
            'startTime'
        ]

    async def is_ready(self, window_size):
        pass

    async def size(self):
        pass

    async def get_last_time(self):
        # Get the last end time 
        pass

    async def get_first_time(self):
        # Get the last end time 
        pass

    async def get_state_for_assets(
        self,
        assets,
        window_size,
        feature_num
    ):

        m = np.random.rand(
            feature_num,
            len(assets),
            window_size
        ).tolist()

        return m

    async def get_state(
        self,
        asset_num,
        window_size,
        feature_num,
        selection_period,
        selection_method="s2vol"
    ):
        
        m = np.random.rand(
            feature_num,
            asset_num,
            window_size
        ).tolist()

        a = ["TEST"+str(i) for i in range(asset_num)]

        return m, a

    async def get_top_assets(
        self,
        asset_num, 
        start_time,
        end_time,
        selection_method='s2vol'
    ):
        top_assets = ["TEST"+str(i) for i in range(asset_num)]

        return top_assets

    async def get_last_price(self, symbol):
        return 1

    async def has_symbol_info(self, symbol):
        return True