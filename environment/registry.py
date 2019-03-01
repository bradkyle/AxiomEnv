from flask import Flask, request, jsonify
import uuid
import numpy as np
import six
import argparse
import sys
import json
from server.server import InvalidUsage
from environment.sandbox import SandboxEnv


import logging
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)

########## Container for environments ##########
class Registry(object):
    """
    Container and manager for the environments instantiated
    on this server.
    When a new environment is created, such as with
    envs.create('CartPole-v0'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """
    def __init__(self, buffer):
        self.envs = {}
        self.id_len = 8
        self.buffer = buffer

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def _remove_env(self, instance_id):
        try:
            del self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    async def create(
        self,
        quote_asset,
        commission,
        feature_num,
        asset_num,
        window_size,
        selection_period,
        selection_method,
        balance_init,
        env_type
        ):
        try:
            if env_type == "sandbox":
                env = SandboxEnv(
                    buffer=self.buffer,
                    quote_asset=quote_asset,
                    commission=commission,
                    asset_num=asset_num,
                    window_size=window_size,
                    feature_num=feature_num,
                    selection_period=selection_period,
                    selection_method=selection_method,
                    balance_init=balance_init
                )
            elif env_type == "test":
                env = SandboxEnv(
                    buffer=self.buffer,
                    quote_asset=quote_asset,
                    commission=commission,
                    asset_num=asset_num,
                    window_size=window_size,
                    feature_num=feature_num,
                    selection_period=selection_period,
                    selection_method=selection_method,
                    balance_init=balance_init
                )
            else:
                raise InvalidUsage("Invalid env type: "+env_type)
        except Exception as e:
            raise InvalidUsage(str(e))

        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    async def list_all(self):
        return dict([(instance_id, env.quote_asset) for (instance_id, env) in self.envs.items()])

    async def reset(self, instance_id):
        env = self._lookup_env(instance_id)
        pv, ff, a = await env.reset()
        return pv, ff, a

    async def step(self, instance_id, action):
        env = self._lookup_env(instance_id)
        pv, ff, a = await env.step(action)
        return pv, ff, a

    async def step(self, instance_id, action):
        env = self._lookup_env(instance_id)
        pv, ff, a = await env.state(action)
        return pv, ff, a

    async def close(self, instance_id):
        env = self._lookup_env(instance_id)
        await env.close()
        self._remove_env(instance_id)