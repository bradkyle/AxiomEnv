from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf

import collections
from client.client import Client

nest = tf.contrib.framework.nest

class PyProcessExchEnv(object):
  """DeepMind Lab wrapper for PyProcess."""

  def __init__(
    self, 
    config,
    remote_base="http://localhost:5000"
    ):

    self.client = Client(remote_base)
    self.instance_id = client.env_create(config)

  def _reset(self):
    self.client.reset(self.instance_id)

  def _observation(self):
    [
        assets, 
        feature_frame, 
        current_pv, 
        pv_prices, 
        pv_values, 
        tnorm
    ] =  self.client.env_state(self.instance_id)
    return feature_frame, current_pv

  def initial(self):
    self._reset()
    return self._observation()

  def step(self, action):
    [
      assets,
      feature_frame, 
      current_pv, 
      tnorm,
      profit
    ] = client.env_step(
      self.instance_id, 
      action
    )
    return feature_frame, current_pv, profit, False

  def close(self):
    self.client.close(self.instance_id)

  @staticmethod
  def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
    """Returns a nest of `TensorSpec` with the method's output specification."""
    batch_size = constructor_kwargs['config'].get('batch_size', 1)
    feature_num = constructor_kwargs['config'].get('feature_num', 3)
    asset_num = constructor_kwargs['config'].get('asset_num', 50)
    window_size = constructor_kwargs['config'].get('window_size', 90)

    observation_spec = [
        tf.contrib.framework.TensorSpec([batch_size, feature_num, asset_num, window_size], tf.float32),
        tf.contrib.framework.TensorSpec([asset_num], tf.float32)
    ]

    if method_name == 'initial':
      return observation_spec
    elif method_name == 'step':
      return (
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.bool),
          observation_spec,
      )


StepOutputInfo = collections.namedtuple(
  'StepOutputInfo',
  'episode_return episode_step'
)

StepOutput = collections.namedtuple(
  'StepOutput',
  'reward info done feature_frame pv'
)


class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_reward = tf.constant(0.)
      initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0))
      initial_done = tf.constant(True)
      
      initial_feature_frame, initial_pv = self._env.initial()

      initial_output = StepOutput(
          initial_reward,
          initial_info,
          initial_done,
          initial_feature_frame,
          initial_pv
      )

      # Control dependency to make sure the next step can't be taken before the
      # initial output has been read from the environment.
      with tf.control_dependencies(nest.flatten(initial_output)):
        initial_flow = tf.constant(0, dtype=tf.int64)
      initial_state = (initial_flow, initial_info)
      
      return initial_output, initial_state

  def step(self, action, state):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      flow, info = nest.map_structure(tf.convert_to_tensor, state)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([flow]):
        feature_frame, pv, reward, done = self._env.step(action)

      with tf.control_dependencies(nest.flatten(observation)):
        new_flow = tf.add(flow, 1)

      # When done, include the reward in the output info but not in the
      # state for the next step.
      new_info = StepOutputInfo(
        info.episode_return + reward,
        info.episode_step + 1
      )

      # Return the elements, either from x or y, depending on the condition.
      # If both x and y are None, then this operation returns the coordinates 
      # of true elements of condition. The coordinates are returned in a 2-D 
      # tensor where the first dimension (rows) represents the number of true 
      # elements, and the second dimension (columns) represents the coordinates 
      # of the true elements. Keep in mind, the shape of the output tensor can 
      # vary depending on how many true values there are in input. Indices are
      # output in row-major order.
      new_state = new_flow, nest.map_structure(
          lambda a, b: tf.where(done, a, b),
          StepOutputInfo(tf.constant(0.), tf.constant(0)),
          new_info
      )

      output = StepOutput(
        reward, 
        new_info, 
        done, 
        feature_frame,
        pv
      )

      return output, new_state