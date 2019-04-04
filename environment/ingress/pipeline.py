from typing import List, Mapping
import faust

from binance.client import Client
from binance.websockets import BinanceSocketManager

app = faust.App('myapp', broker='kafka://localhost')

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
    pass

klines_topic = app.topic('binance_klines', value_type=Kline)
depth_topic = app.topic('binance_depth', value_type=Depth)

# Binance Producers
async def produce_depth_events():
    client = Client("", "")
    bm = BinanceSocketManager(client)
    kline_key = bm.start_multiplex_socket(
            ["ethbtc@kline_1m"],
            process_kline
    )

async def produce_kline_events():
    client = Client("", "")
    bm = BinanceSocketManager(client)
    kline_key = bm.start_multiplex_socket(
            ["ethbtc@kline_1m"],
            process_kline
    )

async def produce_execution_events():
    client = Client("", "")
    bm = BinanceSocketManager(client)
    kline_key = bm.start_multiplex_socket(
            ["ethbtc@kline_1m"],
            process_kline
    )

async def produce_account_events():
    client = Client("", "")
    bm = BinanceSocketManager(client)
    kline_key = bm.start_multiplex_socket(
            ["ethbtc@kline_1m"],
            process_kline
    )

async def combine():
    pass

# todo create tumbling window

tumbling_table = app.Table(
    'tumbling_table',
    default=FeatureVector
).tumbling(
    10, 
    expires=timedelta(hours=3)
).relative_to_field(
    Kline.event_time
)


async def window():
    pass

# @app.agent(app.topic('kline', 'book'))
# async def topic_merger(in_stream_1):
#     async for value in (in_stream_1 & in_topic_2.stream()):
#         print(value)
#         yield value

        # window, combine


