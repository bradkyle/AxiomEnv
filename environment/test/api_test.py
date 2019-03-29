import requests
import pytest
import unittest
from client.client import Client
from constants.environment import EnvConfig
import numpy as np

DC = EnvConfig(
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

def test_env_create():
    instance_id = client.env_create(DC)
    envs = client.env_list_all()
    assert isinstance(envs, list)
    assert instance_id in client.env_list_all()
    client.env_close(instance_id)
    assert instance_id not in client.env_list_all()

def test_buffer_ready():
    buffer_ready = client.buffer_ready(90)
    assert isinstance(buffer_ready, bool)
    assert buffer_ready == True

def test_buffer_size():
    buffer_size = client.buffer_size()
    # assert isinstance(buffer_size, int)

def test_get_top_assets():
    assets = client.get_top_assets(
        asset_number=5,
        selection_period=90,
        selection_method='s2vol'
    )
    assert isinstance(assets, list)
    assert len(assets) == 5

def test_get_random_assets():
    assets = client.get_random_assets(
        asset_number=DC.asset_num
    )
    assert isinstance(assets, list)
    assert len(assets) == DC.asset_num

def test_get_state():
    state = client.get_state(
        asset_number=DC.asset_num,
        window_size=DC.window_size,
        feature_number=DC.feature_num,
        selection_period=DC.selection_period,
        selection_method=DC.selection_method
    )
    assert np.array(state['feature_frame']).shape == (
        DC.feature_num,
        DC.asset_num,
        DC.window_size
    )
    assert len(state['assets']) == DC.asset_num

def test_get_state_from_assets():
    assets = client.get_random_assets(
        asset_number=DC.asset_num
    )
    state = client.get_state_from_assets(
        window_size=DC.window_size,
        feature_number=DC.feature_num,
        assets=assets
    )
    assert np.array(state['feature_frame']).shape == (
        DC.feature_num,
        DC.asset_num,
        DC.window_size
    )
    assert len(state['assets']) == DC.asset_num

def test_env_reset():
    pass

def test_env_step():
    pass

def test_env_state():
    pass

def test_env_info():
    pass

def test_env_close():
    pass