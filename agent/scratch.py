    # Refers to batch size
    # Seen as though the network is used online in production (without batch inference)
    # a default value of 1 has been given such that batch training can occur without having
    # to feed in a value for the input num during production in either tensorflow serving or
    # google cloud machine learning engine.
    self.input_num = tf.placeholder_with_default(
        1,
        shape=[],
        name="input_num"
    )

    # The current feature vector i.e. 
    # [batch * ["close", "high", "low"] 
    # * asset_number * window_size]
    self.input_tensor = tf.placeholder(
        tf.float32,
        shape=[
            None,
            feature_num,
            asset_num,
            window_size
        ],
        name="input_tensor"
    )

    # The previous portfolio vector equal
    # in length to the number of assets
    self.previous_w = tf.placeholder(
        tf.float32, 
        shape=[
            None, 
            asset_num
        ],
        name="previous_w"
    )

    self._feature_num = feature_num;
    self._asset_num = asset_num
    self._window_size = window_size