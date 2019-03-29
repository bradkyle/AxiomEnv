# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Add the ptdraft folder path to the sys.path list
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import tensorflow as tf
import collections
import contextlib
import functools

from agent.agent import Agent
import environment.client.wrappers as wrappers
import numpy as np
import py_process
import sonnet as snt
import vtrace
from actor import build_actor
from learner import build_learner
import environment.constants.environment as env_const

try:
  import agent.dynamic_batching
  tf.logging.info("Running with dynamic batching")
except tf.errors.NotFoundError as e:
  tf.logging.warning('Running without dynamic batching.')
  tf.logging.error(e)

from six.moves import range

nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_integer('save_checkpoint_secs', 600, '')
flags.DEFINE_integer('save_summaries_secs', 30, '')
flags.DEFINE_integer('log_step_count_steps', 50000, '')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')

# Environment
flags.DEFINE_enum('quote_asset', 'BTC', ['BTC', 'BNB', 'ETH', 'USDT'], # todo randomize
                  'The default quote asset that the environment should use')
flags.DEFINE_float('commission', 0.00075, 'Trading consumption') # todo randomize
flags.DEFINE_integer('feature_num', 3, 'Number of features')
flags.DEFINE_integer('asset_num', 15, 'Number of assets to actively trade')
flags.DEFINE_integer('window_size', 10, 'Size of the historical window')
flags.DEFINE_integer('step_rate', 3, 'Amount of steps the agent takes per minute')
flags.DEFINE_integer('selection_period', 90, 'Period over which assets should be selected')
flags.DEFINE_integer('unroll_length', 10, 'Number of steps an agent takes per episode')
flags.DEFINE_integer('seed', 1, 'Random seed.')

flags.DEFINE_enum(
    'selection_method', 
    's2vol', 
    [
        's2', 
        's2vol', 
        'random', 
        'most_traded'
    ],
    'The method to use to select a set of assets'
)

flags.DEFINE_enum(
    'reward_method',
    'profit',
    [
        'profit',
        'portfolio_value',
        'sharpe_ratio',
        'drawdown'
    ], 
    'The reward method to use'
)

flags.DEFINE_float(
    'balance_init', 
    1.0, 
    'The initial amout of quote asset to start with'
)

flags.DEFINE_enum(
    'env_type', 
    'sandbox', 
    ['sandbox', 'live'], # todo randomize
    'The type of environment to use'
)

# Loss settings.
# todo change to adam
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')


AgentOutput = collections.namedtuple(
    'AgentOutput',
    'action policy_logits baseline'
)


DEFAULT_CONFIG = env_const.EnvConfig(
    quote_asset=FLAGS.quote_asset,
    commission=FLAGS.commission,
    feature_num=FLAGS.feature_num,
    asset_num=FLAGS.asset_num,
    window_size=FLAGS.window_size,
    selection_period=FLAGS.selection_period,
    selection_method=FLAGS.selection_method,
    init_balance=FLAGS.balance_init,
    env_type=FLAGS.env_type,
    step_rate=FLAGS.step_rate
)

def is_single_machine():
  return FLAGS.task == -1


def create_environment(config, is_test=False):
  """Creates an exchange environment wrapped in a `FlowEnvironment`."""

  p = py_process.PyProcess(
    wrappers.PyProcessExchEnv,
    config
  )

  return wrappers.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""
  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs


def train():
  """Train."""

  if is_single_machine():
    tf.logging.info("Running on single machine")
    local_job_device = ''
    shared_job_device = ''
    is_actor_fn = lambda i: True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    filters = []
  else:
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # actors. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'    
    
    # Represents a cluster as a set of
    # "tasks", organized into "jobs".
    # A tf.train.ClusterSpec represents the set of 
    # processes that participate in a distributed 
    # TensorFlow computation. Every tf.train.Server 
    # is constructed in a particular cluster.
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']
    })
    
    server = tf.train.Server(
        cluster, 
        job_name=FLAGS.job_name,
        task_index=FLAGS.task
    )
    
    filters = [shared_job_device, local_job_device]

  # Only used to find the actor output structure.
  with tf.Graph().as_default():
    
    agent = Agent(
        feature_num=FLAGS.feature_num,
        asset_num=FLAGS.asset_num,
        window_size=FLAGS.window_size,
        commission=FLAGS.commission
    )
    
    env = create_environment(DEFAULT_CONFIG)
    
    structure = build_actor(
        agent=agent,
        env=env,
        FLAGS=FLAGS
    )    
    
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]
    shapes = [t.shape.as_list() for t in flattened_structure]

  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/cpu'), \
       pin_global_variables(global_variable_device):
    
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

    # BUILD QUEUE
    # ===========================================================================>
    with tf.device(shared_job_device):
      # A queue implementation that dequeues 
      # elements in first-in first-out order.
      # Creates a queue that dequeues elements
      # in a first-in first-out order.
      # A FIFOQueue has bounded capacity; supports
      # multiple concurrent producers and consumers;
      # and provides exactly-once delivery.
      # A FIFOQueue holds a list of up to capacity 
      # elements. Each element is a fixed-length 
      # tuple of tensors whose dtypes are described 
      # by dtypes, and whose shapes are optionally 
      # described by the shapes argument.
      # If the shapes argument is specified, each 
      # component of a queue element must have the
      # respective fixed shape. If it is unspecified,
      # different queue elements may have different 
      # shapes, but the use of dequeue_many is disallowed.
      queue = tf.FIFOQueue(
          capacity=100, 
          dtypes=dtypes, 
          shapes=shapes, 
          shared_name='buffer'
      )
      
      agent = Agent(
        feature_num=FLAGS.feature_num,
        asset_num=FLAGS.asset_num,
        window_size=FLAGS.window_size,
        commission=FLAGS.commission
      )
        
      if is_single_machine() and 'dynamic_batching' in sys.modules:        
        # For single machine training, we use dynamic batching for improved GPU
        # utilization. The semantics of single machine training are slightly
        # different from the distributed setting because within a single unroll
        # of an environment, the actions may be computed using different weights
        # if an update happens within the unroll.
        old_build = agent._build
        @dynamic_batching.batch_fn
        def build(*args):
          with tf.device('/gpu'):
            return old_build(*args)
        tf.logging.info('Using dynamic batching.')
        agent._build = build


    # BUILD ACTORS
    # ===========================================================================>

    # Todo make better for real time environment
    # Build actors and ops to enqueue their output.
    enqueue_ops = []
    for i in range(FLAGS.num_actors): # TODO change to env configurations
      if is_actor_fn(i):
        
        tf.logging.info('Creating actor with config')
        
        env = create_environment(
            DEFAULT_CONFIG
        )
        
        actor_output = build_actor(
            agent=agent,
            env=env,
            FLAGS=FLAGS
        )

        # Append the actor outputs to the 
        # FIFOQueue above in order to pass
        # the environment outputs and action
        # outputs processed later
        with tf.device(shared_job_device):
          enqueue_ops.append(
              queue.enqueue(
                  nest.flatten(
                      actor_output
                  )
              )
          )

    # ADD QUEUE RUNNER
    # ===========================================================================>

    # If running in a single machine setup, run actors with QueueRunners
    # (separate threads).
    if is_learner and enqueue_ops:
      # Holds a list of enqueue operations for a queue, each to be run in a thread.
      # Queues are a convenient TensorFlow mechanism to compute tensors asynchronously 
      # using multiple threads. For example in the canonical 'Input Reader' setup one 
      # set of threads generates filenames in a queue; a second set of threads read 
      # records from the files, processes them, and enqueues tensors on a second queue;
      # a third set of threads dequeues these input records to construct batches and 
      # runs them through training operations.
      # There are several delicate issues when running multiple threads that way:
      # closing the queues in sequence as the input is exhausted, correctly catching
      # and reporting exceptions, etc.
      # The QueueRunner, combined with the Coordinator, helps handle these issues.
      tf.train.add_queue_runner(
          tf.train.QueueRunner(
              queue, 
              enqueue_ops
          )
      )

    # BUILD LEARNER
    # ===========================================================================>
    if is_learner:
      # Create global step, which is the number
      # of environment frames processed.
      tf.get_variable(
          'num_environment_frames',
          initializer=tf.zeros_initializer(),
          shape=[],
          dtype=tf.int64,
          trainable=False,
          collections=[
              tf.GraphKeys.GLOBAL_STEP,
              tf.GraphKeys.GLOBAL_VARIABLES
          ]
      )

      # Create batch (time major) and recreate structure.
      dequeued = queue.dequeue_many(FLAGS.batch_size)
      dequeued = nest.pack_sequence_as(structure, dequeued)

      def make_time_major(s):
        return nest.map_structure(
            lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), 
            s
        )

      # Make dequeued time major
      dequeued = dequeued._replace(
          env_outputs=make_time_major(dequeued.env_outputs),
          agent_outputs=make_time_major(dequeued.agent_outputs)
      )
      
      with tf.device('/gpu'):
        flattened_output = nest.flatten(dequeued)
        # Using StagingArea allows us to prepare the next batch and send it to
        # the GPU while we're performing a training step. This adds up to 1 step
        # policy lag.
        # Class for staging inputs. No ordering guarantees.
        # A StagingArea is a TensorFlow data structure that
        # stores tensors across multiple steps, and exposes
        # operations that can put and get tensors.
        # Each StagingArea element is a tuple of one or more
        # tensors, where each tuple component has a static 
        # dtype, and may have a static shape.
        # The capacity of a StagingArea may be bounded or
        # unbounded. It supports multiple concurrent producers
        # and consumers; and provides exactly-once delivery.
        # Each element of a StagingArea is a fixed-length tuple
        # of tensors whose dtypes are described by dtypes, and 
        # whose shapes are optionally described by the shapes 
        # argument.
        # If the shapes argument is specified, each component 
        # of a staging area element must have the respective 
        # fixed shape. If it is unspecified, different elements
        # may have different shapes,
        # It can be configured with a capacity in which case 
        # put(values) will block until space becomes available.
        area = tf.contrib.staging.StagingArea(
            [t.dtype for t in flattened_output],
            [t.shape for t in flattened_output]
        )
        
        # Operation to add flattened output from
        # dequeued env outputs with their respective
        # agent outputs
        stage_op = area.put(flattened_output)

        # In this instance structure refers to 
        # the output from build actor above
        data_from_actors = nest.pack_sequence_as(
            structure,
            area.get()
        )

        # Unroll agent on sequence, 
        # create losses and update ops.
        output = build_learner(
          agent=agent,         
          env_outputs=data_from_actors.env_outputs,
          agent_outputs=data_from_actors.agent_outputs,
          FLAGS=FLAGS
        )

    # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)

    config = tf.ConfigProto(
        allow_soft_placement=True, 
        device_filters=filters
    )

    # RUN GRAPH
    # ===========================================================================>

    #Creates a MonitoredSession for training.
    # For a chief, this utility sets proper session 
    # initializer/restorer. It also creates hooks 
    # related to checkpoint and summary saving. 
    # For workers, this utility sets proper session 
    # creator which waits for the chief to 
    # initialize/restore. Please check
    # tf.train.MonitoredSession for more information.
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config,
        hooks=[py_process.PyProcessHook()]
    ) as session:

      tf.logging.info('Commencing training run')

      # If the agent is a learner
      if is_learner:
        # Logging.
        summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

        # Prepare data for first run.
        tf.logging.info('Preparing data for first run')

        session.run_step_fn(
            lambda step_context: step_context.session.run(actor_output)
        )

        # Execute learning and track performance.
        num_env_frames_v = 0

        # 
        # =================================================================>
        while num_env_frames_v < FLAGS.total_environment_frames: 
          done_v, infos_v, num_env_frames_v, _ = session.run(
              output + (stage_op,)
          )

          # TODO add logging and metric storage

      else:
        # Execute actors (they just need to enqueue their output).
        tf.logging.info('Running enqueue ops')
        while True:
          session.run(enqueue_ops)



# TODO run env step
def test(action_set, level_names):
  """Test."""

  level_returns = {level_name: [] for level_name in level_names}
  with tf.Graph().as_default():
    agent = Agent(len(action_set))
    outputs = {}
    for level_name in level_names:
      env = create_environment(DEFAULT_CONFIG)
      outputs[level_name] = build_actor(agent, env, level_name, action_set)

    with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir,
        hooks=[py_process.PyProcessHook()]) as session:
      for level_name in level_names:
        tf.logging.info('Testing level: %s', level_name)
        while True:
          done_v, infos_v = session.run((
              outputs[level_name].env_outputs.done,
              outputs[level_name].env_outputs.info
          ))
          returns = level_returns[level_name]
          returns.extend(infos_v.episode_return[1:][done_v[1:]])

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break

  if FLAGS.level_name == 'dmlab30':
    no_cap = dmlab30.compute_human_normalized_score(level_returns,
                                                    per_level_cap=None)
    cap_100 = dmlab30.compute_human_normalized_score(level_returns,
                                                     per_level_cap=100)
    tf.logging.info('No cap.: %f Cap 100: %f', no_cap, cap_100)


def main(_):
  tf.logging.set_verbosity(tf.logging.DEBUG)

  if FLAGS.mode == 'train':
    train()
  else:
    test()


if __name__ == '__main__':
  tf.app.run()

