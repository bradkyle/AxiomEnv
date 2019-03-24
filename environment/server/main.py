import rethinkdb as r
import argparse

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.server as serve_const
from server.server import app

# TODO change
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a Gym HTTP API server')
    parser.add_argument('-l', '--listen', help='interface to listen to', default=serve_const.SERVE_HOST)
    parser.add_argument('-p', '--port', default=serve_const.SERVE_PORT, type=int, help='port to bind to')

    args = parser.parse_args()
    print('Server starting at: ' + 'http://{}:{}'.format(args.listen, args.port))
    app.run(
        host=args.listen,
        port=args.port
    )