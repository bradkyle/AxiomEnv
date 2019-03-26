import rethinkdb as r
from binance.client import Client
import constants.db as db_const
import constants.fields as fields

def flush(conn, storage_period):
    client = Client("", "")
    time_res = client.get_server_time()
    try:
        if time_res['serverTime'] > 2:
            print("Deleting old records")
            cutoff_time = (time_res['serverTime']/1000) - storage_period

            tables = r.table_list().run(conn);

            if db_const.FEATURES_TABLE in tables:
                r.table(db_const.FEATURES_TABLE)\
                .filter(
                    r.row[fields.EPOCH_EVENT_TIME] < r.epoch_time(cutoff_time)
                )\
                .delete()\
                .run(conn)

            if db_const.TRADES_TABLE in tables:
                r.table(db_const.TRADES_TABLE)\
                .filter(
                    r.row[fields.EPOCH_EVENT_TIME] < r.epoch_time(cutoff_time)
                )\
                .delete()\
                .run(conn)

            if db_const.EXECUTION_TABLE in tables:
                r.table(db_const.EXECUTION_TABLE)\
                .filter(
                    r.row[fields.EPOCH_EVENT_TIME] < r.epoch_time(cutoff_time)
                )\
                .delete()\
                .run(conn)

            if db_const.BALANCES_TABLE in tables:
                r.table(db_const.BALANCES_TABLE)\
                .filter(
                    r.row[fields.EPOCH_EVENT_TIME] < r.epoch_time(cutoff_time)
                )\
                .delete()\
                .run(conn)
    except Exception as e:
        print(e)