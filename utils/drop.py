import rethinkdb as r

def drop(conn,db):
        r.db_drop(db)\
        .run(conn)

