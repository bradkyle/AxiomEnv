import sonnet as snt
import tensorflow as tf
import tflearn
import collections

nest = tf.contrib.framework.nest

AgentOutput = collections.namedtuple(
    'AgentOutput',
    'action policy_logits baseline'
)

class Agent(snt.AbstractModule):
    """
    Object representing the actor network, which approximates the function:
        u(s) -> a
    where u (actually mew) is the deterministic policy mapping from states s to
    actions a.
    """
    def __init__(
        self,
        feature_num,
        asset_num,
        window_size,
        commission
    ):  
        super(Agent, self).__init__(name='agent');
        self.feature_num = feature_num
        self.asset_num = asset_num
        self.window_size = window_size
        self.commission = commission
        
    # TODO change to sonnet
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame, prev_w = env_output

        return AgentOutput(
            last_action, 
            tf.expand_dims(tf.zeros([self.asset_num+1], dtype=tf.float32),0), 
            tf.zeros([1], dtype=tf.float32)
        )


    def _build(self, input_):
        action, env_output = input_

        actions, env_outputs = nest.map_structure(
            lambda t: tf.expand_dims(t, 0),
            (action, env_output)
        )

        print("="*70)
        print(actions)
        print("="*70)
        
        outputs = self.unroll(
            actions, 
            env_outputs
        )

        return nest.map_structure(
            lambda t: tf.squeeze(t, 0), 
            outputs
        )


    @snt.reuse_variables
    def unroll(self, actions, env_outputs):
        return snt.BatchApply(self._torso, n_dims=1)((actions, env_outputs))