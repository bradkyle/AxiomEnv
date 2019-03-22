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

        network = tf.expand_dims(frame, 0)

        print("Hello")

        network = tf.transpose(network, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network / network[:, :, -1, 0, None, None]        
        
        with tf.variable_scope('convnet'):
            # ConvLayer
            # TODO add more layers
            network = tflearn.layers.conv_2d(
                incoming=network, 
                nb_filter=3,
                filter_size=[1,2],
                strides=[1,1],
                padding="valid",
                activation="relu",
                regularizer=None,
                weight_decay=0.0,
                name="convlayer"
            )                             

            # TODO add residual layers    
            # # Residual block(s).
            # for j in range(num_blocks):
            #   with tf.variable_scope('residual_%d_%d' % (i, j)):
            #     block_input = conv_out
            #     conv_out = tf.nn.relu(conv_out)
            #     conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            #     conv_out = tf.nn.relu(conv_out)
            #     conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            #     conv_out += block_input
                

        # EIIE Dense
        # input: 4-D Tensor [batch, height, width, in_channels].
        # output: 4-D Tensor [batch, new height, new width, nb_filter].
        width = network.get_shape()[2]
        
        #TODO add more layers
        network = tflearn.layers.conv_2d(
            incoming=network, 
            nb_filter=10,
            filter_size=[1, width],
            strides=[1, 1],
            padding="valid",
            activation="relu",
            regularizer="L2",
            weight_decay=5e-09,
            name="eiie_dense"
        )

        # EIIE Output WithW
        width = network.get_shape()[2]
        height = network.get_shape()[1]
        features = network.get_shape()[3]
   
        # print(width)
        # print(height)
        # print(features)
        # print("================================")

        network = tf.reshape(
            network,
            [
                1,
                int(height),
                1, 
                int(width*features)
            ],
            name="frame_reshape"
        )

        w = tf.reshape(
            prev_w,
            [-1, int(height), 1, 1],
            name="prev_w_reshape"
        )

        # Combine the previous portfolio vector
        # With its corresponding feature frame
        # processed representation
        network = tf.concat(
            [network, w], 
            axis=3,
            name="combine_w_network"
        )
        
        # Output convolution
        network = tflearn.layers.conv_2d(
            incoming=network,
            nb_filter=1,
            filter_size=[1, 1],
            padding="valid",
            regularizer="L2",
            weight_decay=5e-08,
            name="eiie_output"
        )

        policy_logits = network[:, 0, 0]

        # Add quote bias to network
        quote_bias = tf.get_variable(
            "quote_bias", 
            [1,1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer
        )

        new_action = tf.concat(
            [quote_bias, policy_logits],
            axis=1
        )

        new_action = tflearn.layers.core.activation(
            network, 
            activation="softmax",
            name='policy_logits'
        )

        baseline = tf.squeeze(
            snt.Linear(1, name='baseline')(policy_logits),
            axis=-1
        )

        return AgentOutput(
            new_action, 
            policy_logits, 
            baseline
        )

    def _build(self, input_):
        action, env_output = input_

        actions, env_outputs = nest.map_structure(
            lambda t: tf.expand_dims(t, 0),
            (action, env_output)
        )
        
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
        return snt.BatchApply(self._torso)((actions, env_outputs))