import sonnet as snt
import tensorflow as tf
import tflearn

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
    def _build(self, inputs, prev_w, input_num):

        # print(inputs[0])
        print("==========================================")

        # inputs = tf.expand_dims(inputs,0)

        network = tf.transpose(inputs, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network / network[:, :, -1, 0, None, None]        
        
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
            weight_decay=0.0
        )                                               

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
            weight_decay=5e-09
        )

        # EIIE Output WithW
        width = network.get_shape()[2]
        height = network.get_shape()[1]
        features = network.get_shape()[3]
   
        print(width)
        print(height)
        print(features)
        print("================================")

        network = tf.reshape(
            network,
            [
                input_num,
                int(height),
                1, 
                int(width*features)
            ]
        )

        w = tf.reshape(
            prev_w,
            [-1, int(height), 1, 1]
        )

        network = tf.concat(
            [network, w], 
            axis=3
        )
        
        network = tflearn.layers.conv_2d(
            incoming=network,
            nb_filter=1,
            filter_size=[1, 1],
            padding="valid",
            regularizer="L2",
            weight_decay=5e-08
        )

        network = network[:, :, 0, 0]

        # TODO change to quote bias
        btc_bias = tf.get_variable(
            "btc_bias", 
            [1, 1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer
        )

        btc_bias = tf.tile(
            btc_bias, 
            [input_num, 1]
        )

        network = tf.concat(
            [btc_bias, network],
            axis=1
        )

        self.voting = network

        network = tflearn.layers.core.activation(
            network, 
            activation="softmax"
        )

        return network
                
    def pure_pc(self):
        c = self.commission_ratio
        w_t = self.future_omega[:self.input_num-1]  # rebalanced
        w_t1 = self.policy[1:self.input_num]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu