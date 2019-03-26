
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import environment.ingress.binance.prices_ingress as binance
import constants.db as db_const
import argparse

if __name__ == '__main__':
    # TODO create binance features table
    parser = argparse.ArgumentParser(description='Run scripts for managing the rethinkdb database')    
    args = parser.parse_args()

    agent = binance.PricesIngress()

    agent.run()