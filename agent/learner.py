import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import tensorflow as tf
import agent.vtrace as vtrace
nest = tf.contrib.framework.nest



def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  # Measures the probability error in discrete classification tasks 
  # in which the classes are mutually exclusive (each entry is in exactly one class).
  # For example, each CIFAR-10 image is labeled with one and only one label: 
  # an image can be a dog or a truck, but not both.
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions,
      logits=logits
  )

  advantages = tf.stop_gradient(
      advantages
  )

  policy_gradient_loss_per_timestep = cross_entropy * advantages
  
  return tf.reduce_sum(policy_gradient_loss_per_timestep)

# TODO check logical consistency!
def compute_policy_gradient_loss_from_normal_distribution(mus, sigmas, actions, advantages):
    
    norm_dist = tf.distributions.Normal(
        loc=mus, 
        scale=sigmas
    )

    advantages = tf.stop_gradient(
      advantages
    )

    log_prob = norm_dist.log_prob(actions)
    log_prob_rank = log_prob.shape.ndims
    policy_gradient_loss_per_timestep = tf.reduce_sum(log_prob, log_prob_rank-1) * advantages

    return tf.reduce_sum(policy_gradient_loss_per_timestep)

def build_learner(agent, env_outputs, agent_outputs, FLAGS):
    """
    Builds the learner loop.

    Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

    Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
    """

    learner_outputs = agent.unroll(
        agent_outputs.action,
        env_outputs
    )

    # Use last baseline value 
    # (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.

    # Applies func to each entry in 
    # structure and returns a new structure.
    agent_outputs = nest.map_structure(
        lambda t: t[1:], 
        agent_outputs
    )

    rewards, infos, done, _, _ = nest.map_structure(
        lambda t: t[1:], 
        env_outputs
    )
    
    learner_outputs = nest.map_structure(
        lambda t: t[:-1], 
        learner_outputs
    )

    # TODO remove reward clipping and replace with normalization
    # with respect total portfolio value
    if FLAGS.reward_clipping == 'abs_one':
        clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif FLAGS.reward_clipping == 'soft_asymmetric':
        squeezed = tf.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

    discounts = tf.to_float(~done) * FLAGS.discounting
    
    # Compute V-trace returns and weights.
    # Note, this is put on the CPU because it's faster than on GPU. It can be
    # improved further with XLA-compilation or with a custom TensorFlow operation.
    with tf.device('/cpu'):
        vtrace_returns = vtrace.from_normal_distribution(
            behavior_policy_mus=agent_outputs.mu,
            behavior_policy_sigmas=agent_outputs.sigma,
            target_policy_mus=learner_outputs.mu,
            target_policy_sigmas=learner_outputs.sigma,
            actions=agent_outputs.action,
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

    # Compute loss as a weighted sum of the baseline loss, the policy gradient
    # loss and an entropy regularization term.
    total_loss = compute_policy_gradient_loss_from_normal_distribution(
        mus=learner_outputs.mu,
        sigmas=learner_outputs.sigma,
        actions=agent_outputs.action,
        advantages=vtrace_returns.pg_advantages
    )

    total_loss += FLAGS.baseline_cost * compute_baseline_loss(
      vtrace_returns.vs - learner_outputs.baseline
    )

    total_loss += FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits
    )

    # Optimization
    num_env_frames = tf.train.get_global_step()

    learning_rate = tf.train.polynomial_decay(
        FLAGS.learning_rate, 
        num_env_frames,
        FLAGS.total_environment_frames, 
        0
    )

    # TODO change to Adam optimizer
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        FLAGS.decay,
        FLAGS.momentum, 
        FLAGS.epsilon
    )

    train_op = optimizer.minimize(
        total_loss
    )
        
    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        num_env_frames_and_train = num_env_frames.assign_add(
            FLAGS.batch_size * FLAGS.unroll_length
        )

    # Adding a few summaries.
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('total_loss', total_loss)
    # tf.summary.histogram('action', agent_outputs.action)

    return done, infos, num_env_frames_and_train