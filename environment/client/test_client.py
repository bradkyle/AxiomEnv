import requests
import six.moves.urllib.parse as urlparse
import json
import os
import numpy as np
import random
from random import choices
from string import ascii_lowercase

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import environment.constants.rest as rest_const
import environment.constants.environment as env_const

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client(object):
    """
    Gym client to interface with gym_http_server
    """
    def __init__(self, remote_base):
        self.config = None

    def buffer_ready(self, window_size):
        return True

    def buffer_size(self):
        return 90

    def env_create(
        self, 
        config
    ):
        self.config = config
        return "dahweufhaweufhafiushf"

    def env_reset(self, instance_id):
        m = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        pv = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        assets = ["".join(choices(ascii_lowercase, k=3)) for _ in range(self.config.asset_num)]

        tnorm=random.uniform(1.5, 1.9)

        return env_const.StateOutput(
            assets=assets, 
            feature_frame=m, 
            current_pv=pv,
            pv_prices=pv,
            pv_values=pv,
            tnorm=tnorm 
        )

    def env_step(self, instance_id, action):
        route = '/envs/{}/step/'.format(instance_id)
        data = {'action': action}
        m = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        pv = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        assets = ["".join(choices(ascii_lowercase, k=3)) for _ in range(self.config.asset_num)]

        tnorm=random.uniform(1.5, 1.9)
        reward=random.uniform(1.5, 1.9)
        done=False

        return env_const.RawStepOutput(
            assets=assets, 
            feature_frame=m, 
            current_pv=pv,
            pv_prices=pv,
            pv_values=pv,
            tnorm=tnorm,
            reward=reward,
            done=done
        )

    def env_state(self, instance_id):
        route = '/envs/{}/state/'.format(instance_id)
        m = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        pv = np.random.rand(
            self.config.feature_num,
            self.config.asset_num,
            self.config.window_size
        ).tolist()

        assets = ["".join(choices(ascii_lowercase, k=3)) for _ in range(self.config.asset_num)]

        tnorm=random.uniform(1.5, 1.9)

        return env_const.StateOutput(
            assets=assets, 
            feature_frame=m, 
            current_pv=pv,
            pv_prices=pv,
            pv_values=pv,
            tnorm=tnorm 
        )

    def env_info(self, instance_id):
        route = '/envs/{}/action_space/'.format(instance_id)
        return self._get_request(route)

    def env_close(self, instance_id):
        route = '/envs/{}/close/'.format(instance_id)
        self._post_request(route, None)

