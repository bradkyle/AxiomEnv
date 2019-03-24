from binance.client import Client
import rethinkdb as r

conn = r.connect(
         host="localhost",
         port=28015,
         db="binance"
)

client = Client("", "")
info = client.get_exchange_info()

r_info = []
for i in info['symbols']:
     i['Id'] = i['symbol']
     r_info.append(i)

r.table("info").insert(
                r_info,
                conflict="replace"
).run(conn)