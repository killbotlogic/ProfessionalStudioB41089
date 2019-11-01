import time

# import math
import gym
# from gym import spaces, logger
# from gym.utils import seeding
from gym.envs.classic_control import rendering

import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from collections import namedtuple
from itertools import count


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Track:
    def __init__(self, a, b, track_length):
        self.begins_at = a
        self.ends_at = b
        self.track_length = track_length

    def __eq__(self, other):
        return self.begins_at == other.begins_at and self.ends_at == other.ends_at

    def __hash__(self):
        return hash((self.begins_at.__hash__(), self.ends_at.__hash__()))

    def get_delta(self):
        delta_x = self.ends_at.x - self.begins_at.x
        delta_y = self.ends_at.y - self.begins_at.y

        return Node(x=delta_x, y=delta_y)

    def get_angle(self):
        delta_x = self.ends_at.x - self.begins_at.x
        delta_y = self.ends_at.y - self.begins_at.y

        # np.arctan(-1) / np.pi
        return np.arctan(delta_y / delta_x)

    def __repr__(self):
        return f'Track: {self.begins_at} -> {self.ends_at}'

    def geom(self):
        geom = rendering.Line(self.begins_at.arr(), self.ends_at.arr())
        geom.set_color(0, 0, 0)
        return geom

    def __contains__(self, station):
        return station in (self.begins_at, self.ends_at)

class Node:
    def __init__(self, x, y, name=None):
        self.name = name
        # self.p = pos
        self.x = x
        self.y = y
        self._tracks = None
        self._nodes = None

    def __eq__(self, other):
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def tracks(self):
        if not self._tracks:
            self._tracks = {x for x in World.tracks if x.begins_at == self or x.ends_at == self}
        return self._tracks

    def arr(self):
        return self.x, self.y

    def geom(self):
        geom = rendering.make_capsule(length=20, width=20)
        geom.set_color(1, 0, 0)
        geom.add_attr(rendering.Transform(translation=self.arr(), rotation=0.0, scale=(1, 1)))
        geom.add_attr(rendering.Transform(translation=(-10.0, 0), rotation=0.0, scale=(1, 1)))  # center the pill

        return geom

    def is_next_to(self, other_track):
        for t in self.tracks():
            if other_track in (t.begins_at, t.ends_at):
                return True
        return False

    def __repr__(self):
        return f'Node: {self.name} ({self.x}, {self.y})'


class Train:

    def __init__(self, track, dist, direction, name):
        self.on_track = track
        self.dist_on_track = dist
        self.direction = direction  # -1 = from end to beginning, 1 = from beginning to end
        self.name = name
        self.speed = 0.1

        width = 45.0
        height = 15.0
        lef, rig, top, bot = -width / 2, width / 2, height / 2, -height / 2

        self.geom = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        self.geom.set_color(0, 1, 0)
        self.translation = rendering.Transform()
        self.geom.add_attr(self.translation)

    def go_to(self, station):
        pass

    def step(self, direction, next_track, speed):
        if next_track in self.on_track.begins_at.tracks() or next_track in self.on_track.ends_at.tracks():

            if self.pos() == next_track.begins_at:
                self.dist_on_track = 0
                self.on_track = next_track
            if self.pos() == next_track.ends_at:
                self.dist_on_track = next_track.track_length
                self.on_track = next_track

        self.dist_on_track += direction * speed

        if self.dist_on_track < 0.0:
            self.dist_on_track = 0.0
        if self.dist_on_track > self.on_track.track_length:
            self.dist_on_track = self.on_track.track_length

        # if self.pos() == self.on_track.begins_at:
        #     if next_track in self.on_track.begins_at.tracks():
        #
        #         if self.pos() == next_track.begins_at:
        #             self.dist_on_track = 0
        #         else:
        #             self.dist_on_track = next_track.track_length
        #         self.on_track = next_track
        # if self.pos() == self.on_track.ends_at:
        #     if next_track in self.on_track.ends_at.tracks():
        #         if self.pos() == next_track.begins_at:
        #             self.dist_on_track = 0
        #         else:
        #             self.dist_on_track = next_track.track_length
        #         self.on_track = next_track

    def pos(self) -> Node:
        delta = self.on_track.get_delta()
        progress = self.dist_on_track / self.on_track.track_length

        x = self.on_track.begins_at.x + progress * delta.x
        y = self.on_track.begins_at.y + progress * delta.y

        return Node(x=x, y=y)

    def curr_station(self):
        stations = [x for x in World.nodes if x == self.pos()]
        if stations:

            return stations[0]
        else:
            return None

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()




class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    from typing import List, Set
    tracks: Set[Track] = set()
    nodes: List[Node] = []
    points = set()

    def __init__(self):
        self.age = 0
        self.viewer = None
        self.state = None

        self.done = False

        self.scr_wid = 900
        self.scr_hgt = 700

        self.train = None
        self.train_trans = None

        self.origin = None
        self.destination = None

    def step(self, next_station: Node):

        if self.done:
            raise Exception('test is done.')

        curr_station = [x for x in World.nodes if x == self.train.pos()][0]

        if curr_station.is_next_to(next_station):
            track = [x for x in World.tracks if curr_station in x and next_station in x][0]
            if track.ends_at == next_station:
                dir = 1
            else:
                dir = -1
            self.train.step(dir, track, track.track_length)

            self.state = {
                'current_station': next_station
            }


        # self.train.step(direction=action['direction'], next_track=action['next_track'], speed=action['speed'])
        self.age += 1

        self.done = self.destination == self.train.pos()
        if self.done:
            reward = 0.0
        else:
            # reward = 1.0 / self.age
            reward = -1.0

        return self.state['current_station'], reward, self.done, {}

    def reset(self):

        track = list(self.origin.tracks())[0]

        if track.ends_at == self.origin:
            self.train = Train(track=track, dist=track.track_length, direction=1, name='bob')
        else:
            self.train = Train(track=track, dist=0.0, direction=1, name='bob')

        self.state = {
            'current_station': self.origin
        }

        self.done = False

        self.close()
        self.render()
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.scr_wid, self.scr_hgt)

            for track in World.tracks:
                self.viewer.add_geom(track.geom())

            for n in World.nodes:
                self.viewer.add_geom(n.geom())

            self.viewer.add_geom(self.train.geom)

        self.train.translation.set_rotation(self.train.on_track.get_angle())
        self.train.translation.set_translation(self.train.pos().x, self.train.pos().y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


gym.envs.registration.register(
    id='world-v0',
    entry_point='main:World',
)

if __name__ == '__main__':

    from main import *
    import gym
    import numpy as np

    from torch.optim.rmsprop import RMSprop

    na = Node(name='A', x=304.0, y=256.0)
    nb = Node(name='B', x=539.0, y=365.0)

    tb = Node(name='Bravo', x=841.0, y=154.0)
    tc = Node(name='Charlie', x=204.0, y=526.0)
    td = Node(name='Dingo', x=786.0, y=617.0)
    tf = Node(name='Foxtrot', x=56.0, y=285.0)
    tt = Node(name='Tango', x=89.0, y=66.0)
    tw = Node(name='Whiskey', x=249.0, y=64.0)

    ra = Track(tc, td, 5)
    rb = Track(td, tb, 5)

    rc = Track(tb, nb, 3)
    rd = Track(td, nb, 3)
    re = Track(tc, nb, 3)

    rf = Track(na, nb, 3)

    rg = Track(na, tf, 2)
    rh = Track(na, tt, 2)
    ri = Track(na, tw, 2)

    World.nodes = [na, nb, tb, tc, td, tf, tt, tw]
    World.tracks = {ra, rb, rc, rd, re, rf, rg, rh, ri}

    world: World = gym.make('world-v0')

    world.origin = tw
    world.destination = td

    world.reset()
    world.render()


    # =================================================

    RENDER_FPS = 20

    N_INPUTS = len(World.nodes) * 2  # which station now

    N_OUTPUTS = len(World.nodes)  # which station next
    LEARNING_RATE = 0.1
    N_EPISODES = 20  # number of training iterations
    N_MAX_STEPS = 1000  # max steps per episode

    N_GAMES_PER_EPISODE = 3  # train the policy every 10 episodes

    # BATCH_SIZE = 128
    # GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    # TARGET_UPDATE = 10

    # save_iterations = 10  # save the model every 10 training iterations
    DISCOUNT_RATE = 0.95

    BATCH_SIZE = 128
    GAMMA = 0.999

    # 2. Build the neural network

    class NextStationNet(nn.Module):
        def __init__(self):
            super(NextStationNet, self).__init__()
            self.input_layer = nn.Linear(N_INPUTS, 32)

            self.hidden_layer = nn.Linear(32, 32)
            # self.hidden_layer.weight = torch.nn.Parameter(torch.tensor([[1.58]]))
            # self.hidden_layer.bias = torch.nn.Parameter(torch.tensor([-0.14]))

            self.output_layer = nn.Linear(32, N_OUTPUTS)
            # self.output_layer.weight = torch.nn.Parameter(torch.tensor([[2.45]]))
            # self.output_layer.bias = torch.nn.Parameter(torch.tensor([-0.11]))

        def forward(self, x):
            x = torch.sigmoid(self.input_layer(x))
            x = torch.sigmoid(self.hidden_layer(x))
            x = torch.sigmoid(self.output_layer(x))
            return x

    policy_net: NextStationNet = NextStationNet()
    target_net: NextStationNet = NextStationNet()  # DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer: RMSprop = RMSprop(policy_net.parameters())

    steps_done = 0  # super global variable

    def select_action(policy_net, from_station, to_station):
        global steps_done
        # track instance to one hot
        in_arr = torch.tensor([0.0] * N_INPUTS)
        in_arr[World.nodes.index(from_station)] = 1.0
        in_arr[len(World.nodes) + World.nodes.index(to_station)] = 1.0

        # global steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        with torch.no_grad():
            prediction = policy_net(in_arr)
        if random.random() > eps_threshold:

            max_idx = prediction.max(0)[1]
        else:
            max_idx = random.randrange(N_OUTPUTS)

        target = torch.tensor([0.0] * N_OUTPUTS)
        target[max_idx] = 1.0
        station = World.nodes[max_idx]

        return station, prediction, target

    print(f"network topology: {policy_net}")

    # # run input data forward through network
    # # track instance to one hot
    # input_data = torch.tensor([0.0] * N_INPUTS)
    # input_data[World.nodes.index(world.train.curr_station())] = 1.0  # start at node A
    #
    # output = policy_net(input_data)
    #
    #
    # # backpropagate gradient
    #
    # target = torch.tensor([0] * N_OUTPUTS)
    # target[World.nodes.index(td)] = 1  # I want to go to station dingo
    # # target = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    #
    # criterion = nn.MSELoss()
    # loss = criterion(output, target)
    # policy_net.zero_grad()
    # loss.backward()
    #
    # # update weights and biases
    # optimizer = optim.SGD(policy_net.parameters(), lr=0.1)
    # optimizer.step()
    #
    # output = policy_net(input_data)
    # print(f"updated_a_l2 = {round(output.item(), 4)}")


    def discount_rewards(reward, discount_rate):
        discounted_rewards = np.empty(len(reward))
        cumulative_rewards = 0
        for step in reversed(range(len(reward))):
            cumulative_rewards = reward[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards


    def discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [discount_rewards(reward, discount_rate) for reward in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

        # exploration policy. Try something new or stick with the known.

    for episode in range(N_EPISODES):
        print(f"episode: {episode}")
        all_rewards = []  # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        # all_entropy = []
        for game in range(N_GAMES_PER_EPISODE):
            print(f"  game: {game}")

            curr_rewards = []  # all raw rewards from the current episode
            curr_gradients = []  # all gradients from the current episode
            world.reset()

            # world.render()
            # time.sleep(1 / RENDER_FPS)
            for step in range(N_MAX_STEPS):
                next_station, prediction, target = select_action(policy_net, from_station=world.train.curr_station(), to_station=world.destination)
                print(next_station)
                curr_station, reward, done, info = world.step(next_station=next_station)

                world.render()
                time.sleep(1 / RENDER_FPS)

                curr_rewards += [reward]
                # curr_gradients += [val_grads]

                if done:
                    break

            # loss = F.smooth_l1_loss(prediction, target)

            # Optimize the model
            # optimizer.zero_grad()
            # loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            # optimizer.step()

            all_rewards += [curr_rewards]
            print(f"    reward: {len(curr_rewards)}")
            all_gradients += [curr_gradients]

        # At this point we have run the policy for 10 episodes, and we are
        # ready for a policy update using the algorithm described earlier.
        all_rewards_normalized = discount_and_normalize_rewards(all_rewards, DISCOUNT_RATE)
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

    world.step(next_station=na)
    world.render()
    time.sleep(1)
    world.step(next_station=nb)
    world.render()
    time.sleep(1)
    world.step(next_station=td)
    world.render()
    time.sleep(1)