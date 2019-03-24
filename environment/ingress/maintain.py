
import argparse
import rethinkdb as r
import threading

# Add the ptdraft folder path to the sys.path list
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.db as db_const
from utils.create import create
from utils.drop import drop
from utils.flush import flush

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scripts for managing the rethinkdb database')
    parser.add_argument('-r', '--run', help='The method to run', default="")
    parser.add_argument('-l', '--host', help='The rethinkdb host to connect to', default=db_const.DB_HOST)
    parser.add_argument('-p', '--port', help='The rethinkdb port to connect to', default=db_const.DB_PORT)
    parser.add_argument('-d', '--db', help='The rethinkdb database to connect to', default=db_const.DB_NAME)
    parser.add_argument('-c', '--cutoff', help='The cutoff period for old records', default=db_const.CUTOFF_PERIOD)

    args = parser.parse_args()

    conn = r.connect(
        host=args.host,
        port=args.port,
        db=args.db
    )

    # Flush old records from the database
    thread = threading.Thread(target=flush(), args=(conn, args.cutoff))
    thread.daemon = True
    if not thread.is_alive():
        thread.start()


    