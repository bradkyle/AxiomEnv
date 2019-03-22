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

        network = tflearn.layers.conv_2d(
                incoming=frame, 
                nb_filter=3,
                filter_size=[1,2],
                strides=[1,1],
                padding="valid",
                activation="relu",
                regularizer=None,
                weight_decay=0.0,
                name="convlayer"
        ) 

        network = network[:, :, 0, 0]

        policy_logits = snt.Linear(
            self.asset_num+1, 
            name='policy_logits'
        )(network)

        print(policy_logits.get_shape())

        return AgentOutput(
            policy_logits, 
            policy_logits,
            tf.zeros(last_action.get_shape()[0], dtype=tf.float32)
        )


    def _build(self, input_):
        action, env_output = input_
        
        outputs = self._torso((
            action, 
            env_output
        ))

        return outputs

    @snt.reuse_variables
    def unroll(self, actions, env_outputs):
        return snt.BatchApply(self._torso, n_dims=2)((actions, env_outputs))