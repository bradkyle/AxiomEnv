import rethinkdb as r
import numpy as np
import xarray
import pandas as pd
import json
import time
import asyncio
import constants.db as db_const
import constants.fields as fields
from pandas.io.json import json_normalize

# TODO data normalization!
# TODO rand 50
# 

class Buffer():
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
        self.conn = r.connect(
                 host=r_host,
                 port=r_port,
                 db=r_db
        )
        self.feature_table = r_feature_table
        self.prices_table = r_prices_table
        self.quote_asset = quote_asset

        self.index_columns = [
            'baseAsset',
            'startTime'
        ]

        self.whitelisted_features = [
            'close',
            'open',
            'high',
            'low',
            'ask_price_0',
            'ask_price_1',
            'ask_price_2',
            'ask_price_3',
            'ask_price_4',
            'ask_quantity_0',
            'ask_quantity_1',
            'ask_quantity_2',
            'ask_quantity_3',
            'ask_quantity_4',
            'bid_price_0',
            'bid_price_1',
            'bid_price_2',
            'bid_price_3',
            'bid_price_4',
            'bid_quantity_0',
            'bid_quantity_1',
            'bid_quantity_2',
            'bid_quantity_3',
            'bid_quantity_4',
            'trades',
            'volume'
        ]

        self.whitelisted_columns = [
            'baseAsset',
            'startTime'            
        ] + self.whitelisted_features

    def convert_df_to_list(self, df):
        xr = df.to_xarray()

        values = xr.to_array()\
                 .values

        return values.tolist()

    async def is_ready(self, window_size):
        size = await self.size()
        if size >= int(window_size)+1:
            return True
        else:
            return False

    async def size(self):
        last_time = await self.get_last_time()
        first_time = await self.get_first_time()
        return (last_time - first_time)/60

    async def get_last_time(self):
        # Get the last end time 
        last_event_time = r.table(self.feature_table)\
            .max('epochEndTime')\
            .pluck('endTime')\
            .run(self.conn)

        last_time = (last_event_time['endTime']+1)/1000

        return last_time

    async def get_first_time(self):
        # Get the last end time 
        first_event_time = r.table(self.feature_table)\
            .min('epochStartTime')\
            .pluck('startTime')\
            .run(self.conn)

        first_time = (first_event_time['startTime']+1)/1000

        return first_time

    async def get_frame_complex(self, start, last_time, assets):
                # Create an index for the time range
        time_index = pd.to_datetime(list(range(start, round(last_time), 60)),unit='s')
        time_index = time_index.round('min')

        # Get the klines for this period
        rec = r.table(self.feature_table)\
        .between(
            r.epoch_time(start),
            r.epoch_time(last_time),
            index='epochEventTime'
        )\
        .pluck(
            self.whitelisted_columns
        )\
        .filter(lambda doc:
            r.expr(assets)
                .contains(doc["baseAsset"])
        )\
        .run(self.conn)
        
        f = pd.DataFrame(rec)
        f['startTime'] = pd.to_datetime(f['startTime'], unit='ms')
        f.set_index(self.index_columns, inplace=True)
        
        # Reindex the dataframe based upon 
        ind = pd.MultiIndex.from_product(
            [
                f.index.levels[0],
                time_index
            ],
            names=self.index_columns
        )    
        f = f.reindex(ind)
        
        # Fill non existent values
        f=f.fillna(axis=0, method="ffill")\
           .fillna(axis=0, method="bfill")
        
        return f

    async def get_frame(self, start, last_time, assets):
                # Create an index for the time range
        time_index = pd.to_datetime(list(range(start, round(last_time), 60)),unit='s')
        time_index = time_index.round('min')

        # Get the klines for this period
        rec = r.table(self.feature_table)\
        .between(
            r.epoch_time(start),
            r.epoch_time(last_time),
            index='epochEventTime'
        )\
        .pluck(
            'baseAsset',
            'startTime',
            'close',
            'open',
            'high',
            'low'
        )\
        .filter(lambda doc:
            r.expr(assets)
                .contains(doc["baseAsset"])
        )\
        .run(self.conn)
        
        f = pd.DataFrame(rec)
        f['startTime'] = pd.to_datetime(f['startTime'], unit='ms')
        f.set_index(self.index_columns, inplace=True)
        
        # Reindex the dataframe based upon 
        ind = pd.MultiIndex.from_product(
            [
                f.index.levels[0],
                time_index
            ],
            names=self.index_columns
        )    
        f = f.reindex(ind)
        
        # Fill non existent values
        f=f.fillna(axis=0, method="ffill")\
           .fillna(axis=0, method="bfill")
        
        return f

    async def get_state_for_assets(
        self,
        assets,
        window_size,
        feature_num
    ):
        last_time = await self.get_last_time()

        # Derive the start time as a function of the 
        # end time minus the window size
        start = round(last_time - (int(window_size)*60))
        
        f = await self.get_frame(
            start, 
            last_time,
            assets
        )
        
        f = f[['close','high','low']]
        m = self.convert_df_to_list(f)

        return m

    async def get_state(
        self,
        asset_num,
        window_size,
        feature_num,
        selection_period,
        selection_method="s2vol"
    ):
        last_time = await self.get_last_time()

        # Derive the start time as a function of the 
        # end time minus the window size
        start = round(last_time - (int(window_size)*60))
        selection_start = round(last_time - (int(selection_period)*60))
        
        assets = await self._get_top_assets(
            int(asset_num),
            selection_start,
            last_time,
            selection_method
        )

        f = await self.get_frame(
            start, 
            last_time,
            assets
        )

        f = f[['close','high','low']]
        m = self.convert_df_to_list(f)
        a = f.index.levels[0].tolist()
        # TODO make sure order is the same
        return m, a

    async def get_random_assets(
        self,
        asset_num
    ):
        return r.table(self.prices_table)\
                .sample(int(asset_num))\
                .pluck(fields.BASE_ASSET)\
                .map(lambda a:a[fields.BASE_ASSET])\
                .run(self.conn)

    async def get_top_assets(
        self,
        asset_num,
        selection_period,
        selection_method
    ):
        last_time = await self.get_last_time()

        # Derive the start time as a function of the 
        # end time minus the window size
        selection_start = round(last_time - (int(selection_period)*60))

        assets = await self._get_top_assets(
            int(asset_num),
            selection_start,
            last_time,
            selection_method
        )

        return assets

    async def _get_top_assets(
        self,
        asset_num, 
        start_time,
        end_time,
        selection_method='s2vol'
    ):
        q = r.table(self.feature_table)\
        .between(
            r.epoch_time(start_time),
            r.epoch_time(end_time),
            index='epochEventTime'
        )

        if selection_method == "s2vol":
            taq = self.get_top_assets_by_s2vol(
                q,
                asset_num
            )
        elif selection_method == "s2":
            taq = self.get_top_assets_by_s2vol(
                q,
                asset_num
            )
        else:
            raise ValueError("Selection method not valid")
        
        top_assets = taq.run(self.conn)

        return top_assets

    def get_top_assets_by_s2vol(
        self,
        q,
        asset_num
    ):
        return q.group('baseAsset')\
        .map((r.row['close'], r.row['quoteAssetVolume'])).map(lambda x: {
            'count': 1,
            'sum': x[0],
            'vol': x[1],
            'diff': 0 
        })\
        .reduce(lambda a, b: {
            'count': r.add(a['count'], b['count']),
            'sum': r.add(a['sum'], b['sum']),
            'vol': r.add(a['vol'], b['vol']),
            'diff': r.add(
                a['diff'],
                b['diff'],
                r.do(
                r.sub(a['sum'].div(a['count']), b['sum'].div(b['count'])),
                r.div(a['count'].mul(b['count']), a['count'].add(b['count'])),
                lambda avgdelta, weight: r.mul(avgdelta, avgdelta, weight)
                )
            )
        })\
        .ungroup()\
        .map(lambda g: {
            'asset': g['group'],
            'count': g['reduction']['count'],
            'sum': g['reduction']['sum'],
            'vol': g['reduction']['vol'],
            's2': r.branch(g['reduction']['count'].gt(1), r.div(g['reduction']['diff'], g['reduction']['count'].sub(1)), 0)
        })\
        .merge(lambda d: r.do(
            r.div(d['sum'], d['count']),
            r.mul(d['vol'], d['s2']),
            lambda avg, s2vol: {
            'avg': avg,
            's2vol': s2vol,
        }))\
        .order_by(r.desc('s2vol'))\
        .limit(asset_num)\
        .pluck('asset')\
        .map(lambda a:a['asset'])

    def get_top_assets_by_s2(
        self,
        q, 
        asset_num
    ):
        return q.group('base_asset')\
        .map(r.row['close']).map(lambda x: {
        'count': 1,
        'sum': x,
        'min': x,
        'max': x,
        'diff': 0 # M2,n:  sum((val-mean)^2)
        }).reduce(lambda a, b: {
        'count': r.add(a['count'], b['count']),
        'sum': r.add(a['sum'], b['sum']),
        'min': r.branch(a['min'].lt(b['min']), a['min'], b['min']),
        'max': r.branch(a['max'].gt(b['max']), a['max'], b['max']),
        'diff': r.add(
            a['diff'],
            b['diff'],
            r.do(
            r.sub(a['sum'].div(a['count']), b['sum'].div(b['count'])),
            r.div(a['count'].mul(b['count']), a['count'].add(b['count'])),
            lambda avgdelta, weight: r.mul(avgdelta, avgdelta, weight)
            )
        )
        }).ungroup().map(lambda g: {
            'asset': g['group'],
            'count': g['reduction']['count'],
            'sum': g['reduction']['sum'],
            'min': g['reduction']['min'],
            'max': g['reduction']['max'],
            's2': r.branch(g['reduction']['count'].gt(1), r.div(g['reduction']['diff'], g['reduction']['count'].sub(1)), 0)
            }).merge(lambda d: r.do(
            r.div(d['sum'], d['count']),
            lambda avg: {
            'avg': avg,
        }))\
        .order_by(r.desc('s2'))\
        .limit(asset_num)\
        .pluck('asset')\
        .map(lambda a:a['asset'])

    async def get_features(self, start, last_time, assets):
        rec = r.table(self.feature_table)\
        .between(
            r.epoch_time(start),
            r.epoch_time(last_time),
            index=fields.EPOCH_EVENT_TIME
        )\
        .pluck(
            fields.BASE_ASSET,
            fields.QUOTE_ASSET,
            fields.CLOSE,
            fields.OPEN,
            fields.HIGH,
            fields.LOW,
            fields.VOLUME,
            fields.TRADES
        )\
        .filter(lambda doc:
            r.expr(assets)
                .contains(doc[fields.BASE_ASSET])
        )\
        .run(self.conn)

    async def get_normalised_features(self, start, last_time, assets):
        rec = r.table(self.feature_table)\
        .between(
            r.epoch_time(start),
            r.epoch_time(last_time),
            index=fields.EPOCH_EVENT_TIME
        )\
        .pluck(
            fields.BASE_ASSET,
            fields.QUOTE_ASSET,
            fields.CLOSE,
            fields.OPEN,
            fields.HIGH,
            fields.LOW,
            fields.VOLUME,
            fields.TRADES
        )\
        .filter(lambda doc:
            r.expr(assets)
                .contains(doc[fields.BASE_ASSET])
        )\
        .run(self.conn)

    async def get_last_price(self, symbol):
        res = r.table('prices')\
        .get(symbol)\
        .pluck('price')\
        .values()\
        .run(self.conn)
        return res[0]

    async def has_symbol_info(self, symbol):
        return r.table('info')\
        .get_all(symbol)\
        .count()\
        .eq(1)\
        .run(self.conn)

    async def derive_features_from_num(
        self,
        feature_num
    ):
        if feature_num == 1:
            return ['close']
        elif feature_num == 2:
            pass
        elif feature_num == 3:
            return ['close', 'high', 'low']
        elif feature_num == 4:
            return ['close', 'high', 'low', 'open']
        elif feature_num == 5:
            pass
        elif feature_num == 6:
            pass
        elif feature_num == 7:
            pass
        elif feature_num == 8:
            pass
        elif feature_num == 9:
            pass
        elif feature_num == 10:
            pass
        elif feature_num == 11:
            pass
        elif feature_num == 12:
            pass
        elif feature_num == 13:
            pass
        elif feature_num == 14:
            pass
        elif feature_num == 15:
            pass
            