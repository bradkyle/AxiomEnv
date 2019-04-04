from typing import List, Mapping
import faust
import asyncio
from datetime import timedelta

from binance.client import Client
from binance.websockets import BinanceSocketManager

from autobahn.asyncio.websocket import WebSocketClientProtocol, \
    WebSocketClientFactory

app = faust.App('myapp', broker='kafka://localhost:9092')

class Kline(faust.Record, serializer='json'):
    Id: str
    exchange: str
    event_id: list
    event_time: int
    start_time: int
    end_time: int
    symbol: str
    base_asset: str
    quote_asset: str
    interval: str
    o: float
    h: float
    l: float
    c: float
    volume: float
    trades: int
    quote_asset_volume: float
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float
    is_final: bool

class DepthLevel(faust.Record, serializer='json'):
    level: int
    price: float
    quantity: float
    norm_quantity: float

class Depth(faust.Record, serializer='json'):
    Id: str
    symbol: str
    quote_asset: str
    base_asset: str
    levels: int
    asks: List[DepthLevel]
    bids: List[DepthLevel]

class FeatureVector(faust.Record, serializer='json'):
    Id: List[str]
    o: float
    h: float
    l: float
    c: float
    volume: float
    trades: float
    quote_asset_volume: float
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float


klines_topic = app.topic('binance_klines', value_type=Kline)
depth_topic = app.topic('binance_depth', value_type=Depth)

def derive_base_asset(quote_asset, symbol):
        return symbol.replace(quote_asset, '')

# async def process_kline_events(msg):
#     d = msg['data']
#     k = d['k']
#     await klines_topic.send(
#         value=Kline(
#             Id=[k['s'], k['t']],
#             exchange='binance',
#             event_id=['binance', 'feature', d['s'], d['E']],
#             event_time=d['E'],
#             start_time=k['t'],
#             end_time=k['T'],
#             symbol=k['s'],
#             base_asset="",
#             quote_asset="",
#             interval=k['i'],
#             o=float(k['o']),
#             h=float(k['h']),
#             l=float(k['l']),
#             c=float(k['c']),
#             volume=float(k['v']),
#             trades=int(k['n']),
#             quote_asset_volume=float(k['q']),
#             taker_buy_base_asset_volume=float(k['V']),
#             taker_buy_quote_asset_volume=float(k['Q']),
#             is_final=k['x']
#         ),
#     )

# @app.task
# async def produce_kline_events():
#     client = Client("", "")
#     bm = BinanceSocketManager(client)
#     kline_key = bm.start_multiplex_socket(
#             ["ethbtc@kline_1m"],
#             process_kline_events
#     )
#     bm.start()


tumbling_table = app.Table(
    'feature_table',
    default=FeatureVector
).tumbling(
    size=timedelta(seconds=10), 
    expires=timedelta(hours=3)
).relative_to_field(
    Kline.event_time
)

@app.agent(klines_topic)
async def combine_persist(klines):
    async for kline in klines:
        print(kline)


# # todo create tumbling window



if __name__ == '__main__':
    app.main()
