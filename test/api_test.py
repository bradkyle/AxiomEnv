import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import requests
import six.moves.urllib.parse as urlparse
import time
import pytest

from threading import Thread

import server.server as http_server
import client.client as http_client

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########## CONFIGURATION ##########

host = '127.0.0.1'
port = '5000'
def get_remote_base():
    return 'http://{host}:{port}'.format(host=host, port=port)

def setup_background_server():
    def start_server(app):
        app.run(host=host, port=int(port))

    global server_thread
    server_thread = Thread(target=start_server,
                       args=(http_server.app,))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.25) # give it a moment to settle
    logger.info('Server setup complete')

def teardown_background_server():
    route = '/v1/shutdown/'
    headers = {'Content-type': 'application/json'}
    requests.post(urlparse.urljoin(get_remote_base(), route),
                  headers=headers)
    server_thread.join() # wait until teardown happens
    logger.info('Server teardown complete')

def with_server(fn):
    return fn

needs_api_key = pytest.mark.skipif(os.environ.get('OPENAI_GYM_API_KEY') is None, reason="needs OPENAI_GYM_API_KEY")


########## TESTS ##########

##### Valid use cases #####

DEFAULT_CONFIG = {
        'quote_asset': 'BTC',
        'commission': 0.00075,
        'feature_num':3,
        'asset_num':2,
        'window_size': 90,
        'selection_period': 90,
        'selection_method': 's2vol',
        'balance_init':1,
        'env_type': 'test'
}

@with_server
def test_create_destroy():
    client = http_client.Client(get_remote_base())
    instance_id = client.env_create(DEFAULT_CONFIG)
    assert instance_id in client.env_list_all()
    client.env_close(instance_id)
    assert instance_id not in client.env_list_all()

@with_server
def test_reset():
    client = http_client.Client(get_remote_base())

    instance_id = client.env_create(DEFAULT_CONFIG)
    init_obs = client.env_reset(instance_id)


    instance_id = client.env_create(DEFAULT_CONFIG)
    init_obs = client.env_reset(instance_id)
    

@with_server
def test_step():
   client = http_client.Client(get_remote_base())

   instance_id = client.env_create(DEFAULT_CONFIG)
   client.env_reset(instance_id)
   feature_frame, pv, assets = client.env_step(instance_id, [0.5, 0.5])
   assert len(assets) == 2

   instance_id = client.env_create(DEFAULT_CONFIG)
   client.env_reset(instance_id)
   feature_frame, pv, assets = client.env_step(instance_id, [0.5, 0.5])


##### API usage errors #####

@with_server
def test_bad_instance_id():
    ''' Test all methods that use instance_id with an invalid ID'''
    client = http_client.Client(get_remote_base())
    try_these = [lambda x: client.env_reset(x),
                 lambda x: client.env_step(x, [0.5, 0.5]),
                 lambda x: client.env_close(x)]
    for call in try_these:
        try:
            call('bad_id')
        except http_client.ServerError as e:
            assert 'Instance_id' in e.message
            assert e.status_code == 400
        else:
            assert False

@with_server
def test_missing_param_env_id():
    ''' Test client failure to provide JSON param: env_id'''
    class BadClient(http_client.Client):
        def env_create(self, config):
            route = '/envs/'
            data = {} # deliberately omit config
            resp = self._post_request(route, data)
            instance_id = resp['instance_id']
            return instance_id
            
    client = BadClient(get_remote_base())
    try:
        client.env_create(DEFAULT_CONFIG)
    except http_client.ServerError as e:
        assert 'not provided' in e.message
        assert e.status_code == 400
    else:
        assert False

@with_server
def test_missing_param_action():
    ''' Test client failure to provide JSON param: action'''
    class BadClient(http_client.Client):
        def env_step(self, instance_id, action):
            route = '/envs/{}/step/'.format(instance_id)
            data = {} # deliberately omit action
            resp = self._post_request(route, data)
            observation = resp['observation']
            reward = resp['reward']
            done = resp['done']
            info = resp['info']
            return [observation, reward, done, info]
    client = BadClient(get_remote_base())

    instance_id = client.env_create(DEFAULT_CONFIG)
    print(instance_id)
    client.env_reset(instance_id)
    try:
        client.env_step(instance_id, 1)
    except http_client.ServerError as e:
        assert 'action' in e.message
        assert e.status_code == 400
    else:
        assert False

##### Gym-side errors #####

@with_server
@pytest.mark.skip(reason="Need to fix")
def test_create_malformed():
    client = http_client.Client(get_remote_base())
    try:
        client.env_create('bad string')
    except http_client.ServerError as e:
        assert 'malformed environment ID' in e.message
        assert e.status_code == 400
    else:
        assert False

# @with_server
# def test_missing_API_key():
#    client = http_client.Client(get_remote_base())
#    cur_key = os.environ.get('OPENAI_GYM_API_KEY')
#    os.environ['OPENAI_GYM_API_KEY'] = ''
#    try:
#        print 'UPLOADING'
#        print cur_key
#        client.upload('tmp')
#        print '*****'
#    except requests.HTTPError, e:
#        assert e.response.status_code == 400
#    else:
#        assert False
#    finally:
#        if cur_key:
#            os.environ['OPENAI_GYM_API_KEY'] = cur_key
