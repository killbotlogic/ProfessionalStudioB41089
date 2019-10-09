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
        self.dist_on_track += direction * speed
        if self.dist_on_track < 0.0:
            self.dist_on_track = 0.0
        if self.dist_on_track > self.on_track.track_length:
            self.dist_on_track = self.on_track.track_length

        if self.pos() == self.on_track.begins_at:
            if next_track in self.on_track.begins_at.tracks():

                if self.pos() == next_track.begins_at:
                    self.dist_on_track = 0
                else:
                    self.dist_on_track = next_track.track_length
                self.on_track = next_track
        if self.pos() == self.on_track.ends_at:
            if next_track in self.on_track.ends_at.tracks():
                if self.pos() == next_track.begins_at:
                    self.dist_on_track = 0
                else:
                    self.dist_on_track = next_track.track_length
                self.on_track = next_track

    def pos(self):
        delta = self.on_track.get_delta()
        progress = self.dist_on_track / self.on_track.track_length

        x = self.on_track.begins_at.x + progress * delta.x
        y = self.on_track.begins_at.y + progress * delta.y

        return Node(x=x, y=y)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()


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

    def __repr__(self):
        return f'Node: {self.name} ({self.x}, {self.y})'


class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    tracks = set()
    nodes = set()
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

    def step(self, action):
        if self.done:
            raise Exception('test is done.')
        self.train.step(direction=action['direction'], next_track=action['next_track'], speed=action['speed'])
        self.age += 1

        # self.done = self.train.destination == self.train.pos()
        if not self.done:
            reward = 0.0
        else:
            reward = 1.0 / self.age
        return self.state, reward, self.done, {}

    def reset(self):
        self.state = None
        self.done = False
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

    World.nodes = {na, nb, tb, tc, td, tf, tt, tw}
    World.tracks = {ra, rb, rc, rd, re, rf, rg, rh, ri}

    world = gym.make('world-v0')

    world.train = Train(track=rd, dist=3.0, direction=-1, name='bob')
    world.reset()
    world.render()

    world.step(action={
        'direction': -1,
        'next_track': ra,  # re
        'speed': 1.0
    })
    world.render()


    render_fps = 20

    for x in range(100000000):
        world.step(None)

        time.sleep(1 / render_fps)
