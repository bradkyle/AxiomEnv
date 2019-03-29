import tensorflow as tf
import sonnet as snt
nest = tf.contrib.framework.nest
import collections
import sys
import numpy as np

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 
    [
        'env_outputs',
        'agent_outputs'
    ]
)

def build_actor(agent, env, FLAGS):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  initial_action = tf.zeros(FLAGS.asset_num+1, dtype=tf.float32)

  dummy_agent_output = agent((
       tf.expand_dims(initial_action,0),
       nest.map_structure(
           lambda t: tf.expand_dims(t, 0),
           initial_env_output
       )
  ))

  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype),
      dummy_agent_output
  )

  # All state that needs to persist across training iterations. This includes
  # the last environment output and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state,
      (
          initial_env_state,
          initial_env_output, 
          initial_agent_output
      )
  )

  # Run the unroll. `read_value()` is 
  # needed to make sure later usage will
  # return the first values and not a new
  # snapshot of the variables.
  first_values = nest.map_structure(
      lambda v: v.read_value(), 
      persistent_state
  )

  first_env_state, first_env_output, first_agent_output = first_values

  def step(input_, unused_i):
        env_state, env_output, agent_output = input_

        batched_env_output = nest.map_structure(
            lambda t: tf.expand_dims(t, 0),
            env_output
        )

        # TODO update
        agent_output = agent((
            agent_output.action, 
            batched_env_output
        ))

        # TODO remove first element of array in tensor
        env_output, env_state = env.step(
            agent_output[0], 
            env_state
        )

        return env_state, env_output, agent_output

  output = tf.scan(
      step, 
      tf.range(FLAGS.unroll_length), 
      first_values
  )

  _, env_outputs, agent_outputs = output

  # Update persistent state with the
  # last output from the loop.
  assign_ops = nest.map_structure(
        lambda v, t: v.assign(t[-1]),
        persistent_state, 
        output
  )

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent output.
    first_agent_output = nest.map_structure(
        lambda t: t[0], 
        first_agent_output
    )

    agent_outputs = nest.map_structure(
        lambda t: t[:, 0], 
        agent_outputs
    )

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output),
        (
            agent_outputs, 
            env_outputs
        )
    )

    output = ActorOutput(
        env_outputs=full_env_outputs, 
        agent_outputs=full_agent_outputs
    )

    # No backpropagation should be done here.
    return nest.map_structure(
        tf.stop_gradient, 
        output
    )
