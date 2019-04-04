from binance.client import Client
from binance.websockets import BinanceSocketManager
import copy
import rethinkdb as r

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.fields as fields
from ingress import Ingress 

class KlinesIngress(Ingress):
    def __init__(
        self,
        quote_asset='BTC',
        levels=20,
        interval='1m',
        topic="klines"
    ):
        Ingress.__init__(
            self,
            topic=topic
        )

        self.client = Client("", "")
        self.bm = BinanceSocketManager(self.client)

        self.quote_asset = quote_asset
        self.kline_ws_suffix = "@kline_"+str(interval)

    def _maintain(self):
        prev_kline_ws_list = copy.copy(self.kline_ws_list)
        self.set_conf()
        if set(prev_kline_ws_list) != set(self.kline_ws_list):
            self.reset()
        print("Maintainance complete")

    def _setup(self):
        self.set_conf()
        self.kline_key = self.bm.start_multiplex_socket(
            self.kline_ws_list,
            self.process_kline
        )

    def _stop(self):
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
                return True
                
        except Exception as e:
            print(e)
            return False
    
    def derive_base_asset(self, quote_asset, symbol):
        return symbol.replace(quote_asset, '')

    def check_quote_asset(self, quote_asset, symbol):
        return symbol.endswith(quote_asset);

    def process_kline(self, msg):
        d = msg['data']
        k = d['k']

        try:
            # TODO check depth not empty
            kline = {
                fields.ID: [k['s'], k['t']],
                fields.EXCHANGE: 'binance',
                fields.EVENT_ID: ['binance', 'feature', d['s'], d['E']],
                fields.EVENT_TIME: d['E'],                    
                fields.START_TIME: k['t'],
                fields.END_TIME: k['T'],
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
                fields.IS_FINAL: k['x']
            };

            print(kline)

            self.publish(
                key=k['s'],
                value=kline
            )


        except Exception as e:
            print(e)


if __name__ == "__main__":
    worker = KlinesIngress()
    worker.start()