from binance.client import Client
from binance.websockets import BinanceSocketManager
import threading
import time
import copy
import rethinkdb as r
import argparse
from enum import Enum
import argparse

# Add the ptdraft folder path to the sys.path list
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.db as db_const
import constants.fields as fields

class Binance():
    def __init__(
        self,
        quote_asset,
        levels=20,
        interval="1m",
        maintain_every=360,
        r_host=db_const.DB_HOST,
        r_port=db_const.DB_PORT,
        r_user=db_const.DB_USER,
        r_pass=db_const.DB_PASS,
        r_db=db_const.DB_NAME
    ):
        self.ASKS = {};
        self.BIDS  = {};

        self.client = Client("", "")
        self.bm = BinanceSocketManager(self.client)

        self.conn = r.connect(
                 host=r_host,
                 port=r_port,
                 db=r_db
        )
        self.db = r_db
        self.host = r_host
        self.port = r_port

        self.quote_asset = quote_asset
        self.depth_ws_suffix = "@depth"+str(levels)
        self.kline_ws_suffix = "@kline_"+str(interval)

        self.interval = interval
        self.maintain_every = maintain_every

        self.set_conf()

        thread = threading.Thread(target=self.maintain, args=())
        thread.daemon = True                            # Daemonize thread
        if not thread.is_alive():
            thread.start()


    # TODO add try catch
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

    def maintain(self):
        while True:
            time.sleep(self.maintain_every)
            prev_kline_ws_list = copy.copy(self.kline_ws_list)
            prev_depth_ws_list = copy.copy(self.depth_ws_list)
            self.set_conf()
            if set(prev_depth_ws_list) != set(self.depth_ws_list) or\
               set(prev_kline_ws_list) != set(self.kline_ws_list):
               self.reset()
            print("Maintainance complete")

    def process_info(self, info):
        symbols_info = info["symbols"]
        # for symbol_info in symbols_info:
            
        # {
        #     "symbol":"ETHBTC",
        #     "status":"TRADING",
        #     "baseAsset":"ETH",
        #     "baseAssetPrecision":8,
        #     "quoteAsset":"BTC",
        #     "quotePrecision":8,
        #     "orderTypes":["LIMIT","LIMIT_MAKER","MARKET","STOP_LOSS_LIMIT","TAKE_PROFIT_LIMIT"],
        #     "icebergAllowed":true,
        #     "filters":[
        #         {"filterType":"PRICE_FILTER","minPrice":"0.00000000","maxPrice":"0.00000000","tickSize":"0.00000100"},
        #         {"filterType":"PERCENT_PRICE","multiplierUp":"10","multiplierDown":"0.1","avgPriceMins":5},
        #         {"filterType":"LOT_SIZE","minQty":"0.00100000","maxQty":"100000.00000000","stepSize":"0.00100000"},
        #         {"filterType":"MIN_NOTIONAL","minNotional":"0.00100000","applyToMarket":true,"avgPriceMins":5},
        #         {"filterType":"ICEBERG_PARTS","limit":10},
        #         {"filterType":"MAX_NUM_ALGO_ORDERS","maxNumAlgoOrders":5}
        #     ]
        # }
        pass

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

    def process_trade(self):
        pass

    def get_depth(self, data):
        bids = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['bids']]
        asks = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['asks']]
        return bids, asks

    def setup(self):
        self.depth_key = self.bm.start_multiplex_socket(
            self.depth_ws_list,
            self.process_depth
        )
        self.kline_key = self.bm.start_multiplex_socket(
            self.kline_ws_list,
            self.process_kline
        )

    def start(self):
        self.bm.start()

    def stop(self):
        print("Stopping...")
        self.bm.stop_socket(self.depth_key)
        self.bm.stop_socket(self.kline_key)

    def run(self):
        self.setup()
        self.start()

    def reset(self):
            self.stop()
            self.run()

if __name__ == '__main__':
    # TODO create binance features table
    parser = argparse.ArgumentParser(description='Run scripts for managing the rethinkdb database')    
    parser.add_argument('-q', '--quote', help='The quote asset to aggregate data for', default='BTC')
    args = parser.parse_args()

    agent = Binance(args.quote)

    agent.run()