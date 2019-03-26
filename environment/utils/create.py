import constants.db as db_const
import constants.fields as fields
import rethinkdb as r

def create(conn, db):
        try:
                r.db_create(db).run(conn)

                r.db(db).table_create(db_const.FEATURES_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.PRICES_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.DEPTH_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.TRADES_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.INFO_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.EXECUTION_TABLE, primary_key=fields.ID).run(conn)
                r.db(db).table_create(db_const.BALANCES_TABLE, primary_key=fields.ID).run(conn)

                # Create a secondary index on the last_name attribute
                r.table(db_const.FEATURES_TABLE).index_create(fields.BASE_ASSET).run(conn)
                r.table(db_const.FEATURES_TABLE).index_create(fields.EPOCH_EVENT_TIME).run(conn)
                r.table(db_const.FEATURES_TABLE).index_create(fields.EPOCH_START_TIME).run(conn)
                r.table(db_const.FEATURES_TABLE).index_create(fields.EPOCH_END_TIME).run(conn)

                r.table(db_const.PRICES_TABLE).index_create(fields.BASE_ASSET).run(conn)
                r.table(db_const.PRICES_TABLE).index_create(fields.QUOTE_ASSET).run(conn)
                r.table(db_const.PRICES_TABLE).index_create(fields.SYMBOL).run(conn)

                r.table(db_const.TRADES_TABLE).index_create(fields.BASE_ASSET).run(conn)
                r.table(db_const.TRADES_TABLE).index_create(fields.QUOTE_ASSET).run(conn)
                r.table(db_const.TRADES_TABLE).index_create(fields.SYMBOL).run(conn)
                r.table(db_const.TRADES_TABLE).index_create(fields.EPOCH_EVENT_TIME).run(conn)

                r.table(db_const.DEPTH_TABLE).index_create(fields.BASE_ASSET).run(conn)
                r.table(db_const.DEPTH_TABLE).index_create(fields.QUOTE_ASSET).run(conn)
                r.table(db_const.DEPTH_TABLE).index_create(fields.SYMBOL).run(conn)

                r.table(db_const.INFO_TABLE).index_create(fields.BASE_ASSET).run(conn)
                r.table(db_const.INFO_TABLE).index_create(fields.QUOTE_ASSET).run(conn)
                r.table(db_const.INFO_TABLE).index_create(fields.SYMBOL).run(conn)
        except Exception as e:
                print(e)
