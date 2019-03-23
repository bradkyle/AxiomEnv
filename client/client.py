import requests
import six.moves.urllib.parse as urlparse
import json
import os

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import constants.rest as rest_const
import constants.environment as env_const

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
        return self._get_request(route)

    def buffer_size(self):
        route = '/buffer/size/'
        return self._get_request(route)

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
        return self._post_request(route, data)

    def get_state_from_assets(
        self,
        window_size,
        feature_number,
        assets
    ):
        route = '/state/{ws}/{fn}/{sp}/{sm}'
        route = route.format(
            ws=window_size,
            fn=feature_number
        )
        data = {'assets': assets}
        return self._post_request(route, data)

    def env_create(
        self, 
        config
    ):
        route = '/envs/'

        if not type(config) is env_const.EnvConfig:
            raise ValueError()

        data = config._asdict()
        return self._post_request(route, data)['instance_id']

    def env_list_all(self):
        route = '/envs/'
        return self._get_request(route)['envs']

    def env_reset(self, instance_id):
        route = '/envs/{}/reset/'.format(instance_id)
        resp = self._post_request(route, None)
        return env_const.StateOutput(**resp)

    def env_step(self, instance_id, action):
        route = '/envs/{}/step/'.format(instance_id)
        data = {'action': action}
        resp = self._post_request(route, data)
        return env_const.RawStepOutput(**resp)

    def env_state(self, instance_id):
        route = '/envs/{}/state/'.format(instance_id)
        resp = self._get_request(route)
        return env_const.StateOutput(**resp)

    def env_info(self, instance_id):
        route = '/envs/{}/action_space/'.format(instance_id)
        return self._get_request(route)

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
    
    