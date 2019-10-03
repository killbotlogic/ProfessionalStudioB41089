import time

# import math
import gym
# from gym import spaces, logger
# from gym.utils import seeding
from gym.envs.classic_control import rendering


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


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
        delta_x = self.ends_at.p.x - self.begins_at.p.x
        delta_y = self.ends_at.p.y - self.begins_at.p.y
        return delta_x, delta_y

    def get_angle(self):
        delta_x = self.ends_at.p.x - self.begins_at.p.x
        delta_y = self.ends_at.p.y - self.begins_at.p.y

        # np.arctan(-1) / np.pi
        return np.arctan(delta_y / delta_x)

    def step(self):
        pass


class Train:

    def __init__(self, track, dest, dist, direction, name):
        self.on_track = track
        self.destination = dest
        self.distance_from_beginning_of_track = dist
        self.direction = direction  # -1 = from end to beginning, 1 = from beginning to end
        self.name = name
        self.speed = 0.1

    def go_to(self, station):
        pass

    def step(self):
        self.distance_from_beginning_of_track += self.speed

    def pos(self):
        delta = self.on_track.get_delta()
        progress = self.distance_from_beginning_of_track / self.on_track.track_length

        x = self.on_track.begins_at.p.x + progress * delta[0]
        y = self.on_track.begins_at.p.y + progress * delta[1]

        return x, y

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()


class Node:
    def __init__(self, name, pos):
        self.name = name
        self.p = pos
        self._tracks = None
        self._nodes = None

    def __eq__(self, other):
        return self.p == other.position

    def __hash__(self):
        return self.p.__hash__()

    def tracks(self):
        if not self._tracks:
            self._tracks = {x for x in World.tracks if x.begins_at == self or x.ends_at == self}
        return self._tracks

    def pos(self):
        return self.p.x, self.p.y

    def __repr__(self):
        return f'Node {self.name}'


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


        na = Node('A', Point(x=304.0, y=256.0))
        nb = Node('B', Point(x=539.0, y=365.0))

        tb = Node('Bravo', Point(x=841.0, y=154.0))
        tc = Node('Charlie', Point(x=204.0, y=526.0))
        td = Node('Dingo', Point(x=786.0, y=617.0))
        tf = Node('Foxtrot', Point(x=56.0, y=285.0))
        tt = Node('Tango', Point(x=89.0, y=66.0))
        tw = Node('Whiskey', Point(x=249.0, y=64.0))

        ra = Track(tc, td, 5)
        rb = Track(td, tb, 5)

        rc = Track(tb, nb, 3)
        rd = Track(td, nb, 3)
        re = Track(tc, nb, 3)

        rf = Track(na, nb, 3)

        rg = Track(na, tf, 2)
        rh = Track(na, tt, 2)
        ri = Track(na, tw, 2)

        self.dest_node = nb

        # dest might be superfluous
        self.train = Train(track=re, dest=nb, dist=1.0, direction=-1, name='bob')
        self.train_trans = None
        World.nodes = {na, nb, tb, tc, td, tf, tt, tw}
        World.tracks = {ra, rb, rc, rd, re, rf, rg, rh, ri}

    def step(self, action):
        if self.done:
            raise Exception('test is done.')
        self.train.step()
        self.age += 1

        self.done = self.dest_node.pos == self.train.pos()

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

            for n in World.nodes:
                pretty = rendering.make_capsule(length=20, width=20)
                pretty.set_color(1, 0, 0)
                pretty.add_attr(rendering.Transform(translation=n.pos(), rotation=0.0, scale=(1, 1)))
                self.viewer.add_geom(pretty)

            for track in World.tracks:
                line = rendering.Line(track.begins_at.pos(), track.ends_at.pos())
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)

            cartwidth = 90.0
            cartheight = 30.0
            lef, rig, top, bot = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            train = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
            train.set_color(0, 1, 0)
            self.train_trans = rendering.Transform()
            train.add_attr(self.train_trans)
            self.viewer.add_geom(train)

        #     train_1 = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        #     train_1.set_color(0, 1, 0)
        #     self.train_1_trans = rendering.Transform()
        #     train_1.add_attr(self.train_1_trans)
        #     self.viewer.add_geom(train_1)

        # if self.state is None:
        #     return None
        #
        # state = self.state
        #
        # t0_track = self.tracks[state[0]]
        # t1_track = self.tracks[state[1]]
        # t0_track_loc = t0_track['ctr']
        # t1_track_loc = t1_track['ctr']
        #
        self.train_trans.set_rotation(self.train.on_track.get_angle())
        self.train_trans.set_translation(self.train.pos()[0], self.train.pos()[1])

        # self.train_1_trans.set_rotation(t1_track['rot'])
        #
        # self.train_0_trans.set_translation(t0_track_loc[0], t0_track_loc[1])
        # self.train_1_trans.set_translation(t1_track_loc[0], t1_track_loc[1])

        # self.train_0_trans.set_rotation()

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
    render_fps = 20

    import main
    import gym
    import numpy as np

    world = gym.make('world-v0')

    world.reset()
    world.render()
    for x in range(100000000):
        world.step(None)

        time.sleep(1 / render_fps)
