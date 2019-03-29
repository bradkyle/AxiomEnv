import pytest
import unittest
import asyncio
from core.buffer import Buffer
from constants.environment import EnvConfig
import numpy as np

# TODO check that feature frame is returned in the correct order
# TODO check that feature frame is normalised

DC = EnvConfig(
    quote_asset='BTC',
    commission=0.075,
    feature_num=3,
    asset_num=5,
    window_size=10,
    selection_period=90,
    selection_method='s2vol',
    init_balance=1,
    env_type='sandbox',
    step_rate=3
)

@pytest.mark.asyncio
async def test_get_frame():
    buffer = Buffer()
    last_time = await buffer.get_last_time()

    # Derive the start time as a function of the 
    # end time minus the window size
    start = round(last_time - (int(DC.window_size)*60))

    assets = await buffer.get_random_assets(DC.asset_num)

    f = await buffer.get_frame_complex(
        start, 
        last_time,
        assets
    )

    print(f)





