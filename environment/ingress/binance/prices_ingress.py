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
        self.kline_key = self.bm.start_multiplex_socket(
            self.kline_ws_list,
            self.process_kline
        )

    def _stop(self):
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
                return True
                
        except Exception as e:
            print(e)
            return False

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
        price = {
            fields.ID:k['s'],
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

    def get_depth(self, data):
        bids = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['bids']]
        asks = [{'price':float(l[0]), 'quantity':float(l[1])} for l in data['asks']]
        return bids, asks

if __name__ == '__main__':
    # TODO create binance features table
    parser = argparse.ArgumentParser(description='Run scripts for managing the rethinkdb database')    
    parser.add_argument('-q', '--quote', help='The quote asset to aggregate data for', default='BTC')
    args = parser.parse_args()

    agent = Binance(args.quote)

    agent.run()