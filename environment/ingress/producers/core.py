import threading
import time
import rethinkdb as r
import argparse
from enum import Enum
import argparse

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.db as db_const

class Ingress():
    def __init__(
        self,
        maintain_every=360,
        r_host=db_const.DB_HOST,
        r_port=db_const.DB_PORT,
        r_user=db_const.DB_USER,
        r_pass=db_const.DB_PASS,
        r_db=db_const.DB_NAME
    ):
        self.db = r_db
        self.host = r_host
        self.port = r_port
        self.maintain_every = maintain_every

        self.conn = r.connect(
                 host=r_host,
                 port=r_port,
                 db=r_db
        )
        
    def setup(self):
        self._setup()

    def start(self):
        thread = threading.Thread(target=self.maintain, args=())
        thread.daemon = True
        if not thread.is_alive():
            thread.start()

        self._start()

    def stop(self):
        print("Stopping...")
        self._stop()

    def run(self):
        self.setup()
        self.start()

    def reset(self):
        self.stop()
        self.run()

    def maintain(self):
        while True:
            time.sleep(self.maintain_every)
            self._maintain()

    def _setup(self):
        raise NotImplementedError

    def _maintain(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError

    def _start(self):
        raise NotImplementedError
