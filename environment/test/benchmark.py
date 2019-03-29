import requests
import pytest
import unittest
from client.client import Client
from constants.environment import EnvConfig
import numpy as np
import time

DEFAULT_CONFIG = EnvConfig(
    quote_asset='BTC',
    commission=0.075,
    feature_num=3,
    asset_num=50,
    window_size=90,
    selection_period=90,
    selection_method='s2vol',
    init_balance=1,
    env_type='sandbox',
    step_rate=3
)

client = Client("http://localhost:5000")

def test_performance():
    start = time.time()
    for x in range(1000):
        state = client.get_state(
            asset_number=DEFAULT_CONFIG.asset_num,
            window_size=DEFAULT_CONFIG.window_size,
            feature_number=DEFAULT_CONFIG.feature_num,
            selection_period=DEFAULT_CONFIG.selection_period,
            selection_method=DEFAULT_CONFIG.selection_method
        )
        print(x)
    elapsed_time_lc=(time.time()-start)
    print(elapsed_time_lc)
    assert elapsed_time_lc