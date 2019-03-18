# examples/1_hello/hello.py
import os
import asyncio
from japronto import Application
import constants.rest as rest_const

import logging
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)

from server.exceptions import InvalidUsage, RouteNotFound
from environment.buffer import Buffer
from environment.test_buffer import TestBuffer
from environment.registry import Registry

########## App setup ##########
app = Application()
r = app.router

########## Instantiate globals ##########
env_type = os.environ.get('ENV_TYPE')
if env_type == "production":
    buffer = Buffer()
else:
    buffer = TestBuffer()

registry = Registry(
   buffer=buffer
)

########## Error handling ##########

def get_required_param(json, param):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value=='') or (value==[]):
        logger.info("A required request parameter '{}' had value {}".format(param, value))
        raise InvalidUsage("A required request parameter '{}' was not provided".format(param))
    return value

def get_optional_param(json, param, default):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value=='') or (value==[]):
        logger.info("An optional request parameter '{}' had value {} and was replaced with default value {}".format(param, value, default))
        value = default
    return value

def get_required_param_enum(json, param):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value=='') or (value==[]):
        logger.info("A required request parameter '{}' had value {}".format(param, value))
        raise InvalidUsage("A required request parameter '{}' was not provided".format(param))
    return value

# TODO fix for japronto
def handle_invalid_usage(request, exception):
    return request.Response(
        code=exception.status_code,
        json=exception.to_dict()
    )

# You can also override default 404 handler if you want
def handle_not_found(request, exception):
    return request.Response(
        code=404,
        text=exception.to_dict()
    )

app.add_error_handler(InvalidUsage, handle_invalid_usage)
app.add_error_handler(RouteNotFound, handle_not_found)

########## API general definitions ##########

async def is_buffer_ready(request):
    p = request.match_dict
    ready = await buffer.is_ready(p.window_size)
    return request.Response(json={
        'is_ready':ready
    })

async def buffer_size(request):
    size = await buffer.size()
    return request.Response(json={
        'size':size,
    })

# TODO validate
async def get_state(request):
    p = request.match_dict

    feature_frame, assets = await buffer.get_state_from_assets(
        asset_num=p.an,
        window_size=p.ws,
        feature_num=p.fn,
        selection_period=p.sp,
        selection_method=p.sm
    )

    return request.Response(json={
        'feature_frame':feature_frame,
        'assets': assets
    })

async def get_state_from_assets(request):
    p = request.match_dict

    try:
        json = request.json
    except JSONDecodeError:
        pass

    assets = get_required_param(json, 'assets')

    feature_frame = await buffer.get_state(
        assets=assets,
        window_size=p.ws,
        feature_num=p.fn,
        selection_period=p.sp,
        selection_method=p.sm
    )

    return request.Response(json={
        'feature_frame':feature_frame
    })

########## API environment definitions ##########

async def env_create(request):
    try:
        json = request.json
    except JSONDecodeError:
        pass

    quote_asset = get_required_param(json, 'quote_asset')
    commission = get_required_param(json, 'commission')
    feature_num = get_required_param(json, 'feature_num')
    asset_num = get_required_param(json, 'asset_num')
    window_size = get_required_param(json, 'window_size')
    selection_period = get_required_param(json, 'selection_period')
    selection_method = get_required_param(json, 'selection_method')
    balance_init = get_required_param(json, 'balance_init')
    env_type = get_required_param(json, 'env_type')

    instance_id = await registry.create(
        quote_asset=quote_asset,
        commission=commission,
        feature_num=feature_num,
        asset_num=asset_num,
        window_size=window_size,
        selection_period=selection_period,
        selection_method=selection_method,
        balance_init=balance_init,
        env_type=env_type
    )

    return request.Response(json={
        'instance_id':instance_id
    })

async def env_list_all(request):
    all_envs = await registry.list_all()

    return request.Response(json={
        'envs': all_envs
    })

async def env_reset(request):
    p = request.match_dict

    pv, ff, a = await registry.reset(
        instance_id=p['instance_id']
    )

    return request.Response(json={
        'feature_frame':ff,
        'pv':pv,
        'assets':a
    })

async def env_step(request):
    p = request.match_dict

    try:
        json = request.json
    except JSONDecodeError:
        pass

    action = get_required_param(
        json,
        'action'
    )

    if not isinstance(action, list):
        raise InvalidUsage("Action should be a list");

    [
        assets,
        feature_frame, 
        current_pv, 
        tnorm,
        profit
    ] = await registry.step(
        instance_id=p['instance_id'],
        action=action
    )

    return request.Response(json={
        'feature_frame':ff,
        'pv': cpv, 
        'assets':a
    })

async def env_state(request):
    p = request.match_dict

    [
        assets,
        feature_frame,
        current_pv,
        pv_prices,
        pv_values,
        tnorm
    ] = await registry.state(
        instance_id=p['instance_id']
    )

    return request.Response(json={
        'assets':assets,
        'feature_frame': feature_frame, 
        'current_pv':current_pv,
        'pv_prices':pv_prices,
        'pv_values':pv_values,
        'tnorm': tnorm
    })

async def env_info(request):
    p = request.match_dict

    info = await registry.close(
        instance_id=p['instance_id']
    )

    return request.Response(json={
        'info':info 
    })

async def env_close(request):
    p = request.match_dict

    await registry.close(
        instance_id=p['instance_id']
    )

    return request.Response(code=200)


########## API route definitions ##########

r.add_route('/buffer/ready/{window_size}/', is_buffer_ready, methods=['GET'])
r.add_route('/buffer/size/', is_buffer_ready, methods=['GET'])

r.add_route('/state/{an}/{ws}/{fn}/{sp}/{sm}', get_state, methods=['GET'])
r.add_route('/state/{ws}/{fn}/{sp}/{sm}', get_state_from_assets, methods=['POST'])

r.add_route('/envs/', env_create, methods=['POST'])
r.add_route('/envs/', env_list_all, methods=['GET'])

r.add_route('/envs/{instance_id}/reset/', env_reset, methods=['POST'])
r.add_route('/envs/{instance_id}/step/', env_step, methods=['POST'])
r.add_route('/envs/{instance_id}/state/', env_state, methods=['GET'])
r.add_route('/envs/{instance_id}/info/', env_info, methods=['GET'])
r.add_route('/envs/{instance_id}/close/', env_close, methods=['POST'])




# TODO change
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a Gym HTTP API server')
    parser.add_argument('-l', '--listen', help='interface to listen to', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to bind to')

    args = parser.parse_args()
    print('Server starting at: ' + 'http://{}:{}'.format(args.listen, args.port))
    app.run(
        host=args.listen,
        port=args.port
    )