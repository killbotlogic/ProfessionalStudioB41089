# circular speeding train test
import time

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering


class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):

        # self.seed()
        self.viewer = None
        self.state = None

        self.done = False
        # self.velocity_0 = None
        # self.velocity_1 = None

        def make_circle(n_edges, radius, scr_size):
            tracks = []

            offs = scr_size / 2
            for idx in range(n_edges):
                angle_bgn = (idx + 0.0) / n_edges * np.pi * 2
                angle_ctr = (idx + 0.5) / n_edges * np.pi * 2
                angle_end = (idx + 1.0) / n_edges * np.pi * 2

                bgn_x = np.cos(angle_bgn) * radius + offs
                bgn_y = np.sin(angle_bgn) * radius + offs

                ctr_x = np.cos(angle_ctr) * radius + offs
                ctr_y = np.sin(angle_ctr) * radius + offs

                end_x = np.cos(angle_end) * radius + offs
                end_y = np.sin(angle_end) * radius + offs

                track = {
                    'beg': [bgn_x, bgn_y],
                    'ctr': [ctr_x, ctr_y],
                    'end': [end_x, end_y],
                    'rot': angle_ctr + np.pi / 2
                }
                tracks += [track]
            return tracks

        self.scr_wid = 600
        self.scr_hgt = 600
        self.tracks = make_circle(32, 200, self.scr_wid)

    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.done:
            raise Exception('test is done.')

        state = self.state

        t0_track_id = state[0]
        t1_track_id = state[1]

        t0_track_id += action[0]
        t1_track_id += action[1]

        if t0_track_id == len(self.tracks):
            t0_track_id = 0
        if t1_track_id == len(self.tracks):
            t1_track_id = 0

        self.state = [t0_track_id, t1_track_id]

        self.done = t0_track_id == t1_track_id

        if not self.done:
            reward = 1.0
        else:
            reward = 0.0

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = [0, int(len(self.tracks) / 2)]
        # self.velocity_0 = 0
        # self.velocity_1 = 0
        self.done = False
        return np.array(self.state)

    def render(self, mode='human'):

        cartwidth = 60.0
        cartheight = 30.0

        if self.viewer is None:

            self.viewer = rendering.Viewer(self.scr_wid, self.scr_hgt)

            for track in self.tracks:
                line = rendering.Line(track['beg'], track['end'])
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)

            lef, rig, top, bot = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            train_0 = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
            train_0.set_color(1, 0, 0)
            self.train_0_trans = rendering.Transform()
            train_0.add_attr(self.train_0_trans)
            self.viewer.add_geom(train_0)

            train_1 = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
            train_1.set_color(0, 1, 0)
            self.train_1_trans = rendering.Transform()
            train_1.add_attr(self.train_1_trans)
            self.viewer.add_geom(train_1)

        if self.state is None:
            return None

        state = self.state

        t0_track = self.tracks[state[0]]
        t1_track = self.tracks[state[1]]
        t0_track_loc = t0_track['ctr']
        t1_track_loc = t1_track['ctr']

        self.train_0_trans.set_rotation(t0_track['rot'])
        self.train_1_trans.set_rotation(t1_track['rot'])

        self.train_0_trans.set_translation(t0_track_loc[0], t0_track_loc[1])
        self.train_1_trans.set_translation(t1_track_loc[0], t1_track_loc[1])

        # self.train_0_trans.set_rotation()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




if __name__ == '__main__':
    # import gym
    #
    # gym.envs.registration.register(
    # import tensorflow as tf

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import math
    import gym
    # import time
    import numpy as np
    # import pandas as pd
    # import keras
    # import matplotlib.pyplot as plt
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import random_ops


    def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                     seed=None, dtype=dtypes.float32):
        """Returns an initializer that generates tensors without scaling variance.
        When initializing a deep network, it is in principle advantageous to keep
        the scale of the input variance constant, so it does not explode or diminish
        by reaching the final layer. This initializer use the following formula:
        ```python
          if mode='FAN_IN': # Count only number of input connections.
            n = fan_in
          elif mode='FAN_OUT': # Count only number of output connections.
            n = fan_out
          elif mode='FAN_AVG': # Average number of inputs and output connections.
            n = (fan_in + fan_out)/2.0
            truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
        ```
        * To get [Delving Deep into Rectifiers](
           http://arxiv.org/pdf/1502.01852v1.pdf) (also know as the "MSRA
           initialization"), use (Default):<br/>
          `factor=2.0 mode='FAN_IN' uniform=False`
        * To get [Convolutional Architecture for Fast Feature Embedding](
           http://arxiv.org/abs/1408.5093), use:<br/>
          `factor=1.0 mode='FAN_IN' uniform=True`
        * To get [Understanding the difficulty of training deep feedforward neural
          networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
          use:<br/>
          `factor=1.0 mode='FAN_AVG' uniform=True.`
        * To get `xavier_initializer` use either:<br/>
          `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
          `factor=1.0 mode='FAN_AVG' uniform=False`.
        Args:
          factor: Float.  A multiplicative factor.
          mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
          uniform: Whether to use uniform or normal distributed random initialization.
          seed: A Python integer. Used to create random seeds. See
                `tf.compat.v1.set_random_seed` for behavior.
          dtype: The data type. Only floating point types are supported.
        Returns:
          An initializer that generates tensors with unit variance.
        Raises:
          ValueError: if `dtype` is not a floating point type.
          TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
        """
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
        if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
            raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

        # pylint: disable=unused-argument
        def _initializer(shape, dtype=dtype, partition_info=None):
            """Initializer function."""
            if not dtype.is_floating:
                raise TypeError('Cannot create initializer for non-floating point type.')
            # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
            # This is the right thing for matrix multiply and convolutions.
            if shape:
                fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
                fan_out = float(shape[-1])
            else:
                fan_in = 1.0
                fan_out = 1.0
            for dim in shape[:-2]:
                fan_in *= float(dim)
                fan_out *= float(dim)
            if mode == 'FAN_IN':
                # Count only number of input connections.
                n = fan_in
            elif mode == 'FAN_OUT':
                # Count only number of output connections.
                n = fan_out
            elif mode == 'FAN_AVG':
                # Average number of inputs and output connections.
                n = (fan_in + fan_out) / 2.0
            if uniform:
                # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
                limit = math.sqrt(3.0 * factor / n)
                return random_ops.random_uniform(shape, -limit, limit,
                                                 dtype, seed=seed)
            else:
                # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
                trunc_stddev = math.sqrt(1.3 * factor / n)
                return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                                   seed=seed)

        # pylint: enable=unused-argument

        return _initializer

    def discount_rewards(rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards


    def discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


    def make_car(initializer, learning_rate):
        car = {}
        car['X'] = tf.placeholder(tf.float32, shape=[None, 1])
        car['Hidden'] = tf.layers.dense(inputs=car['X'], units=1, activation=tf.nn.elu, kernel_initializer=initializer)
        car['Logits'] = tf.layers.dense(inputs=car['Hidden'], units=1, kernel_initializer=initializer)  # linear activation
        car['Output'] = tf.nn.sigmoid(car['Logits'])  # probability of right

        car['OutputArray'] = tf.concat(axis=1, values=[1 - car['Output'], car['Output']])
        car['Action'] = tf.multinomial(tf.log(car['OutputArray']), num_samples=1)

        car['Entropy'] = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(car['Action']), logits=car['Logits'])  # AKA: Loss
        optimizer = tf.train.AdamOptimizer(learning_rate)

        car['Gradients'] = []
        car['GradientsAndVariables'] = []
        for grad, variable in optimizer.compute_gradients(car['Entropy']):
            car['Gradients'] += [grad]
            car['GradientsAndVariables'] += [(tf.placeholder(tf.float32, shape=grad.get_shape()), variable)]

        car['Trainer'] = optimizer.apply_gradients(car['GradientsAndVariables'])
        return car

    gym.envs.registration.register(
        id='myenv-v0',
        entry_point='examples.ignorethis:MyEnv',
    )

    # 1. Specify the neural network architecture
    # n_inputs = 1  # == env.observation_space.shape[0]
    # n_hidden_1 = 128  # it's a simple task, we don't need more hidden neurons
    # n_hidden_2 = 128  # it's a simple task, we don't need more hidden neurons
    # n_hidden = 4  # it's a simple task, we don't need more hidden neurons

    n_outputs = 1  # only outputs the probability of accelerating left
    render_fps = 20000
    # learning_rate = 0.01
    learning_rate = 0.1
    n_iterations = 20  # number of training iterations
    n_max_steps = 1000  # max steps per episode
    n_games_per_update = 3  # train the policy every 10 episodes
    # save_iterations = 10  # save the model every 10 training iterations
    discount_rate = 0.95

    # initializer = tf.contrib.layers.variance_scaling_initializer()
    initializer = variance_scaling_initializer()

    # 2. Build the neural network
    car_a = make_car(initializer, learning_rate)
    # car_b = make_car(initializer, learning_rate)

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    env = gym.make('myenv-v0')

    sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print(f"iteration: {iteration}")
        all_rewards = []  # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        all_entropy = []
        for game in range(n_games_per_update):
            print(f"  game: {game}")

            current_rewards = []  # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            obs = env.reset()
            env.render()
            time.sleep(1 / render_fps)
            velocity = 0
            predator_velocity = 1
            for step in range(n_max_steps):

                val_log, val_out, val_act, val_ent, val_grads = sess.run(
                    fetches=[car_a['Logits'], car_a['Output'], car_a['Action'], car_a['Entropy'], car_a['Gradients']],
                    feed_dict={car_a['X']: np.array([[velocity]])})  # one obs
                all_entropy += [val_ent]
                # val_log, val_out, val_act, val_ent, val_grads = sess.run(
                #     fetches=[car_a['Logits'], car_a['Output'], car_a['Action'], car_a['Entropy'], car_a['Gradients']],
                #     feed_dict={car_a['X']: np.array([[velocity]])})  # one obs

                velocity = val_act[0][0]

                obs, reward, done, info = env.step([velocity, predator_velocity])
                predator_velocity = int(not predator_velocity)
                env.render()
                time.sleep(1 / render_fps)

                current_rewards += [reward]
                current_gradients += [val_grads]

                if done:
                    break

            all_rewards += [current_rewards]
            print(f"    reward: {len(current_rewards)}")
            all_gradients += [current_gradients]
        # At this point we have run the policy for 10 episodes, and we are
        # ready for a policy update using the algorithm described earlier.
        all_rewards_normalized = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for idx, grad_and_var in enumerate(car_a['GradientsAndVariables']):
            # multiply the gradients by the action scores, and compute the mean

            # mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index] for game_index, rewards in enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)
            yo = []
            for game_index, rewards in enumerate(all_rewards_normalized):
                for step, reward in enumerate(rewards):
                    yo += [reward * all_gradients[game_index][step][idx]]
            mean_gradients = np.mean(yo, axis=0)
            feed_dict[grad_and_var[0]] = mean_gradients
        # train here
        sess.run(car_a['Trainer'], feed_dict=feed_dict)

    sess.close()
