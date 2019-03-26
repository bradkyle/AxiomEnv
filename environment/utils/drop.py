import rethinkdb as r

def drop(conn,db):
        try:
                r.db_drop(db)\
                .run(conn)
        except Exception as e:
                print(e)

