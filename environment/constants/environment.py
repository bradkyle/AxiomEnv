import collections


EnvConfig = collections.namedtuple(
    'EnvConfig',
    [
      'quote_asset', 
      'commission', 
      'feature_num',
      'asset_num',
      'window_size',
      'selection_period',
      'selection_method',
      'init_balance',
      'env_type',
      'step_rate'
    ]
)

StateOutput = collections.namedtuple(
    'StateOutput',
    [
      'assets', 
      'feature_frame', 
      'current_pv',
      'pv_prices',
      'pv_values',
      'tnorm'
    ]
)

RawStepOutput = collections.namedtuple(
    'StepOutput',
    [
      'assets', 
      'feature_frame', 
      'current_pv',
      'pv_prices',
      'pv_values',
      'tnorm',
      'reward', 
      'done',
    ]
)

StepOutputInfo = collections.namedtuple(
  'StepOutputInfo',
  [
      'episode_return', 
      'episode_step'
  ]
)

StepOutput = collections.namedtuple(
    'StepOutput',
    [
      'reward', 
      'info', 
      'done',
      'feature_frame',
      'pv'
    ]
)