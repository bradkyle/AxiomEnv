from binance.client import Client
from binance.websockets import BinanceSocketManager
import copy
import rethinkdb as r

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.fields as fields
import constants.db as db_const
from ingress.ingress import Ingress 

class FeaturesIngress(Ingress):
    def __init__(
        self,
        quote_asset='BTC',
        levels=20,
        interval='1m'
    ):
        self.ASKS = {};
        self.BIDS  = {};

        self.client = Client("", "")
        self.bm = BinanceSocketManager(self.client)

        self.quote_asset = quote_asset
        self.depth_ws_suffix = "@depth"+str(levels)
        self.kline_ws_suffix = "@kline_"+str(interval)

    def _maintain(self):
        prev_kline_ws_list = copy.copy(self.kline_ws_list)
        prev_depth_ws_list = copy.copy(self.depth_ws_list)
        self.set_conf()
        if set(prev_depth_ws_list) != set(self.depth_ws_list) or\
            set(prev_kline_ws_list) != set(self.kline_ws_list):
            self.reset()
        print("Maintainance complete")

    def _setup(self):
        self.set_conf()
        self.depth_key = self.bm.start_multiplex_socket(
            self.depth_ws_list,
            self.process_depth
        )
        self.kline_key = self.bm.start_multiplex_socket(
            self.kline_ws_list,
            self.process_kline
        )

    def _stop(self):
        self.bm.stop_socket(self.depth_key)
        self.bm.stop_socket(self.kline_key)

    def _start(self):
        self.bm.start()

    # Local
    # =====================================================================>

    def set_conf(self):
        try:
            info = self.client.get_exchange_info()
            linfo = [x['symbol'] for x in info['symbols'] if self.check_quote_asset(self.quote_asset,x['symbol'])]
            if len(linfo) > 5:
                self.kline_ws_list = [y.lower()+self.kline_ws_suffix for y in linfo]
                self.depth_ws_list = [y.lower()+self.depth_ws_suffix for y in linfo]
                return True
                
        except Exception as e:
            print(e)
            return False

    def process_depth(self, msg):
        symbol = msg['stream'].replace(self.depth_ws_suffix, '').upper()
        bids, asks = self.get_depth(msg['data'])
        self.ASKS[symbol] = asks
        self.BIDS[symbol] = bids
    
    def derive_base_asset(self, quote_asset, symbol):
        return symbol.replace(quote_asset, '')

    def check_quote_asset(self, quote_asset, symbol):
        return symbol.endswith(quote_asset);

    def get_asks(self, symbol):
        return self.ASKS[symbol]

    def get_bids(self, symbol):
        return self.BIDS[symbol]

    def process_kline(self, msg):
        d = msg['data']
        k = d['k']
        # TODO check depth not empty
        if d['s'] in self.ASKS:
            kline = {
                'Id': [k['s'], k['t']],
                'eventId': ['binance', 'feature', d['s'], d['E']],
                'eventTime': d['E'],                    
                'startTime': k['t'],
                'endTime': k['T'],
                'epochStartTime': r.epoch_time(k['t']/1000),
                'epochEndTime': r.epoch_time(k['T']/1000),
                'epochEventTime': r.epoch_time(d['E']/1000),
                'symbol': k['s'],
                'baseAsset': self.derive_base_asset(self.quote_asset,k['s']),
                'quoteAsset': self.quote_asset,
                'interval': k['i'],
                'open': float(k['o']),
                'close': float(k['c']),
                'high': float(k['h']),
                'low': float(k['l']),
                'volume': float(k['v']),
                'trades': k['n'],
                'quoteAssetVolume': float(k['q']),
                'takerBuyBaseAssetVolume': float(k['V']),
                'takerBuyQuoteAssetVolume': float(k['Q']),
                'isFinal': k['x'],
                'asks': self.get_asks(k['s']),
                'bids': self.get_bids(k['s']),
            };
            r.table(db_const.FEATURES_TABLE).insert(
                kline,
                conflict="replace"
            ).run(self.conn)

    def get_depth(self, data):
        bids = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['bids']]
        asks = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['asks']]
        return bids, asks