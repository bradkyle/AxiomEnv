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
from ingress.core import Ingress 

class FeaturesIngress(Ingress):
    def __init__(
        self,
        quote_asset='BTC',
        levels=20,
        interval='1m',
        r_host=db_const.DB_HOST,
        r_port=db_const.DB_PORT,
        r_user=db_const.DB_USER,
        r_pass=db_const.DB_PASS,
        r_db=db_const.DB_NAME
    ):
        Ingress.__init__(
            self,
            r_host=r_host,
            r_port=r_port,
            r_user=r_user,
            r_pass=r_pass,
            r_db=r_db
        )
        
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
        print("Starting "+ self.quote_asset)
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

        try:
            # TODO check depth not empty
            if d['s'] in self.ASKS:
                kline = {
                    fields.ID: [k['s'], k['t']],
                    fields.EVENT_ID: ['binance', 'feature', d['s'], d['E']],
                    fields.EVENT_TIME: d['E'],                    
                    fields.START_TIME: k['t'],
                    fields.END_TIME: k['T'],
                    fields.EPOCH_START_TIME: r.epoch_time(k['t']/1000),
                    fields.EPOCH_END_TIME: r.epoch_time(k['T']/1000),
                    fields.EPOCH_EVENT_TIME: r.epoch_time(d['E']/1000),
                    fields.SYMBOL: k['s'],
                    fields.BASE_ASSET: self.derive_base_asset(self.quote_asset,k['s']),
                    fields.QUOTE_ASSET: self.quote_asset,
                    fields.INTERVAL: k['i'],
                    fields.OPEN: float(k['o']),
                    fields.CLOSE: float(k['c']),
                    fields.HIGH: float(k['h']),
                    fields.LOW: float(k['l']),
                    fields.VOLUME: float(k['v']),
                    fields.TRADES: k['n'],
                    fields.QUOTE_ASSET_VOLUME: float(k['q']),
                    fields.TAKER_BUY_BASE_ASSET_VOLUME: float(k['V']),
                    fields.TAKER_BUY_QUOTE_ASSET_VOLUME: float(k['Q']),
                    fields.IS_FINAL: k['x'],
                    fields.ASKS: self.get_asks(k['s']),
                    fields.BIDS: self.get_bids(k['s']),
                };
                r.table(db_const.FEATURES_TABLE).insert(
                    kline,
                    conflict="replace"
                ).run(self.conn)

                price = {
                    fields.ID:k['s'],
                    fields.EVENT_ID: ['binance', 'price', d['s'], d['E']],
                    fields.EVENT_TIME: d['E'],
                    fields.EPOCH_EVENT_TIME: r.epoch_time(d['E']/1000),
                    fields.QUOTE_ASSET: self.quote_asset,
                    fields.BASE_ASSET: self.derive_base_asset(self.quote_asset,k['s']),
                    fields.SYMBOL: k['s'],
                    fields.PRICE: float(k['c'])
                }
                r.table(db_const.PRICES_TABLE).insert(
                    price,
                    conflict="replace"
                ).run(self.conn)
        except Exception as e:
            print(e)

    def get_depth(self, data):
        bids = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['bids']]
        asks = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['asks']]
        return bids, asks

