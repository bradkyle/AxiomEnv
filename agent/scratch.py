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

     while num_env_frames_v < FLAGS.total_environment_frames:
            
          # 
          level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
              (data_from_actors.level_name,) + output + (stage_op,)
          )
        
          # 
          level_names_v = np.repeat(
              [level_names_v],
              done_v.shape[0],
              0
          )

          for level_name, episode_return, episode_step in zip(
              level_names_v[done_v],
              infos_v.episode_return[done_v],
              infos_v.episode_step[done_v]
          ):
            
            episode_frames = episode_step * FLAGS.num_action_repeats

            tf.logging.info(
                'Level: %s Episode return: %f',
                 level_name, 
                 episode_return
            )

            summary = tf.summary.Summary()
            
            summary.value.add(
                tag=level_name + '/episode_return',
                simple_value=episode_return
            )
            
            summary.value.add(
                tag=level_name + '/episode_frames',
                simple_value=episode_frames
            )
            
            summary_writer.add_summary(
                summary, 
                num_env_frames_v
            )

            if FLAGS.level_name == 'dmlab30':
              level_returns[level_name].append(episode_return)

            
          if (FLAGS.level_name == 'dmlab30' and
              min(map(len, level_returns.values())) >= 1):
            
            no_cap = dmlab30.compute_human_normalized_score(
                level_returns,
                per_level_cap=None
            )
            
            cap_100 = dmlab30.compute_human_normalized_score(
                level_returns,
                per_level_cap=100
            )
            
            summary = tf.summary.Summary()
            
            summary.value.add(
                tag='dmlab30/training_no_cap',
                simple_value=no_cap
            )
            
            summary.value.add(
                tag='dmlab30/training_cap_100', 
                simple_value=cap_100
            )
            
            summary_writer.add_summary(
                summary, 
                num_env_frames_v
            )

            # Clear level scores.
            level_returns = {level_name: [] for level_name in level_names}
