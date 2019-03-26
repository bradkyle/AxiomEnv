
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import ingress.binance.features_ingress as binance
import constants.db as db_const
import argparse

if __name__ == '__main__':
    # TODO create binance features table
    parser = argparse.ArgumentParser(description='Run scripts for managing the rethinkdb database')    
    parser.add_argument('-q', '--quote', help='The quote asset to aggregate data for', default='BTC')
    parser.add_argument('-l', '--host', help='The rethinkdb host to connect to', default=db_const.DB_HOST)
    parser.add_argument('-p', '--port', help='The rethinkdb port to connect to', default=db_const.DB_PORT)
    parser.add_argument('-d', '--db', help='The rethinkdb database to connect to', default=db_const.DB_NAME)
    args = parser.parse_args()

    agent = binance.FeaturesIngress(
        args.quote,
        r_host=args.host,
        r_port=args.port,
    )

    agent.run()