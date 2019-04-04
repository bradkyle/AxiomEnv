import sonnet as snt
import tensorflow as tf
import tflearn
import collections

nest = tf.contrib.framework.nest

AgentOutput = collections.namedtuple(
    'AgentOutput',
    [
        'action',
        'policy_logits',
        'baseline',
        'mu',
        'sigma'
    ]
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
                nb_filter=self.feature_num,
                filter_size=[1,2],
                strides=[1,1],
                padding="valid",
                activation="relu",
                regularizer=None,
                weight_decay=0.0,
                name="convlayer"
        )

        network = tflearn.layers.conv_2d(
            frame, 
            2,
            [1, 2],
            [1, 1, 1, 1],
            'valid',
            'relu'
        )

        width = network.get_shape()[2]

        network = tflearn.layers.conv_2d(
            network, 
            48,
            [1, width],
            [1, 1],
            "valid",
            'relu',
            regularizer="L2",
            weight_decay=5e-9
        )

        w_previous=tf.placeholder(tf.float32,shape=[None,self.M])

        network=tf.concat(
            [
                network,
                tf.reshape(w_previous, [-1, self.M, 1, 1])
            ],
            axis=3
        )

        network = tflearn.layers.conv_2d(
            network, 
            1,
            [1, network.get_shape()[2]],
            [1, 1],
            "valid",
            'relu',
            regularizer="L2",
            weight_decay=5e-9
        )

        network=tf.layers.flatten(network)

        w_init = tf.random_uniform_initializer(
            -0.005, 
            0.005
        )

        action = tf.layers.dense(
            network, 
            self.asset_num, 
            activation=tf.nn.softmax, 
            kernel_initializer=w_init
        )

        mu = tf.layers.dense(
            network, 
            self.asset_num, 
            activation=tf.nn.tanh, 
            kernel_initializer=w_init
        )

        sigma = tf.layers.dense(
            network, 
            self.asset_num, 
            activation=tf.nn.softplus, 
            kernel_initializer=w_init
        )

        baseline = tf.squeeze(
            snt.Linear(1, name='baseline')(network), 
            axis=-1
        )

        return AgentOutput(
            action=action, 
            policy_logits=action,
            baseline=baseline,
            mu=mu,
            sigma=sigma
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