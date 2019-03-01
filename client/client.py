import requests
import six.moves.urllib.parse as urlparse
import json
import os
import constants.rest as rest_const

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client(object):
    """
    Gym client to interface with gym_http_server
    """
    def __init__(self, remote_base):
        self.remote_base = remote_base
        self.session = requests.Session()
        self.session.headers.update({'Content-type': 'application/json'})

    def _parse_server_error_or_raise_for_status(self, resp):
        j = {}
        try:
            j = resp.json()
        except:
            # Most likely json parse failed because of network error, not server error (server
            # sends its errors in json). Don't let parse exception go up, but rather raise default
            # error.
            resp.raise_for_status()
        if resp.status_code != 200:  # descriptive message from server side
            print(resp.status_code)
            print(j["message"])
            raise ServerError(message=j["message"], status_code=resp.status_code)
        resp.raise_for_status()
        return j

    def _post_request(self, route, data):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("POST {}\n{}".format(url, json.dumps(data)))
        resp = self.session.post(urlparse.urljoin(self.remote_base, route),
                            data=json.dumps(data))
        return self._parse_server_error_or_raise_for_status(resp)

    def _get_request(self, route):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("GET {}".format(url))
        resp = self.session.get(url)
        return self._parse_server_error_or_raise_for_status(resp)

    def buffer_ready(self, window_size):
        route = '/buffer/ready/{}/'.format(window_size)
        resp = self._get_request(route)
        is_ready = resp['is_ready']
        return is_ready

    def buffer_size(self):
        route = '/buffer/size/'
        resp = self._get_request(route)
        size = resp['size']
        return size

    def get_state(
        self,
        asset_number,
        window_size,
        feature_number,
        selection_period,
        selection_method
    ):
        route = '/state/{an}/{ws}/{fn}/{sp}/{sm}'
        route = route.format(
            an=asset_number,
            ws=window_size,
            fn=feature_number,
            sp=selection_period,
            sm=selection_method
        )
        resp = self._post_request(route, data)
        feature_frame = resp['instance_id']
        assets = resp['assets']
        return assets, feature_frame

    def get_state_from_assets(
        self,
        window_size,
        feature_number,
        selection_period,
        selection_method
    ):
        route = '/state/{ws}/{fn}/{sp}/{sm}'
        route = route.format(
            ws=window_size,
            fn=feature_number,
            sp=selection_period,
            sm=selection_method
        )
        data = {'assets': assets}
        resp = self._post_request(route, data)
        feature_frame = resp['feature_frame']
        return feature_frame

    def env_create(
        self, 
        config
    ):
        route = '/envs/'

        if not type(config) is dict:
            raise ValueError()

        data = {
            'quote_asset': config['quote_asset'],
            'commission': config['commission'],
            'feature_num': config['feature_num'],
            'asset_num': config['asset_num'],
            'window_size': config['window_size'],
            'selection_period': config['selection_period'],
            'selection_method': config['selection_method'],
            'balance_init': config['balance_init'],
            'env_type': config['env_type']
        }
        resp = self._post_request(route, data)
        instance_id = resp['instance_id']
        return instance_id

    def env_list_all(self):
        route = '/envs/'
        resp = self._get_request(route)
        all_envs = resp['envs']
        return all_envs

    def env_reset(self, instance_id):
        route = '/envs/{}/reset/'.format(instance_id)
        resp = self._post_request(route, None)
        feature_frame = resp['feature_frame']
        pv = resp['pv']
        assets = resp['assets']
        return [feature_frame, pv, assets]

    def env_step(self, instance_id, action):
        route = '/envs/{}/step/'.format(instance_id)
        data = {'action': action}
        resp = self._post_request(route, data)
        feature_frame = resp['feature_frame']
        pv = resp['pv']
        assets = resp['assets']
        return [
            assets,
            feature_frame, 
            current_pv, 
            tnorm,
            profit
        ]

    def env_state(self, instance_id):
        route = '/envs/{}/state/'.format(instance_id)
        resp = self._get_request(route)
        pv = resp['pv']
        assets = resp['assets']
        current_pv = resp['current_pv']
        pv_values = resp['pv_values']
        pv_prices = resp['pv_prices']
        feature_frame = resp['feature_frame']
        tnorm = resp['tnorm']
        return [
            assets, 
            feature_frame, 
            current_pv, 
            pv_prices, 
            pv_values, 
            tnorm
        ]

    def env_info(self, instance_id):
        route = '/envs/{}/action_space/'.format(instance_id)
        resp = self._get_request(route)
        info = resp['info']
        return info

    def env_close(self, instance_id):
        route = '/envs/{}/close/'.format(instance_id)
        self._post_request(route, None)

class ServerError(Exception):
    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

if __name__ == '__main__':
    remote_base = ('http://{host}:{port}').format(
        host=rest_const.REST_HOST,
        port=rest_const.REST_PORT
    )
    client = Client(remote_base)

    # Create environment
    instance_id = client.env_create(env_id)

    # Check properties
    all_envs = client.env_list_all()

    # Run a single step
    init_obs = client.env_reset(instance_id)
    [observation, reward, done, info] = client.env_step(
        instance_id,
        1, #TODO change action
    )
    
    