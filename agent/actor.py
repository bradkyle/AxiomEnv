import tensorflow as tf
import sonnet as snt
nest = tf.contrib.framework.nest

def build_actor(agent, env):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  initial_input_num = tf.constant(1)

  dummy_agent_output = agent(
          getattr(initial_env_output, 'feature_frame'),
          getattr(initial_env_output, 'pv'),
          initial_input_num
  )

  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype),
      dummy_agent_output
  )

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
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

  # TODO add pause 
  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_state, agent_output = input_

    action, agent_state = agent(
        env_output,
        agent_state
    )

    # TODO remove first element of array in tensor
    env_output, env_state = env.step(
        raw_action, 
        env_state
    )

    return env_state, env_output, agent_state, agent_output

  # Run the unroll. `read_value()` is 
  # needed to make sure later usage will
  # return the first values and not a new
  # snapshot of the variables.
  first_values = nest.map_structure(
      lambda v: v.read_value(), 
      persistent_state
  )

  first_env_state, first_env_output, first_agent_output = first_values

   # TODO add pause 
  def step(input_, unused_i):
        """Steps through the agent and the environment."""
        env_state, env_output, agent_state, agent_output = input_

        action, agent_state = agent(
            env_output,
            agent_state
        )

        # TODO remove first element of array in tensor
        env_output, env_state = env.step(
            raw_action, 
            env_state
        )

        return env_state, env_output, agent_state, agent_output

  ps = tf.while_loop(
        True,
        body,
        loop_vars,
        shape_invariants=None,
        parallel_iterations=10,
        back_prop=True,
        swap_memory=False,
        name=None,
        maximum_iterations=None,
        return_same_structure=False
  ) 

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_state = nest.map_structure(
        lambda t: t[0],
        first_agent_state
    )

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
        (agent_outputs, env_outputs)
    )

    output = ActorOutput(
        agent_state=first_agent_state,
        env_outputs=full_env_outputs, 
        agent_outputs=full_agent_outputs
    )

    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)
