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


class Train:
    def __init__(self, track, dest, dist, direction, name):
        self.on_track = track
        self.destination = dest
        self.distance_from_beginning_of_track = dist
        self.direction = direction  # -1 = from end to beginning, 1 = from beginning to end
        self.name = name

    def go_to(self, station):
        pass

    def step(self):
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()


class Node:
    def __init__(self, name, pos):
        self.name = name
        self.position = pos
        self.routing_table = RoutingTable()
        self._min_dist = {}
        self._tracks = None
        self._nodes = None

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return self.position.__hash__()

    def tracks(self):
        if not self._tracks:
            self._tracks = {x for x in Earth.tracks if x.begins_at == self or x.ends_at == self}
        return self._tracks

    def nodes(self):
        if not self._nodes:
            self._nodes = set()
            for x in self.tracks():
                if x.begins_at != self:
                    self._nodes |= {x.begins_at}
                    self._min_dist[x.begins_at] = (x.track_length, x.begins_at)
                if x.ends_at != self:
                    self._nodes |= {x.ends_at}
                    self._min_dist[x.ends_at] = (x.track_length, x.ends_at)
        return self._nodes

    def min_dist(self, dest):
        if dest in self._min_dist:
            return self._min_dist[dest]
        import random

        return 9999,

        # try one hop
        # for x in self._nodes:
        #     if x.min_dist(dest):
        #         wx_dist = self._min_dist[x] + x.min_dist(dest)
        #         if dest not in self._min_dist or wx_dist <= self.min_dist(dest):
        #             self._min_dist[dest] = wx_dist
        #     # else:
        #     #     # try two hops
        #     #     for y in x._nodes:
        #     #         if y.min_dist(dest):
        #     #             wxy_dist = self._min_dist[x] + x.min_dist(y) + y.min_dist(dest)
        #     #             xy_dist = x.min_dist(y) + y.min_dist(dest)
        #     #             if dest not in self._min_dist or wxy_dist <= self.min_dist(dest):
        #     #                 self._min_dist[dest] = wxy_dist
        #     #             if dest not in x._min_dist or xy_dist <= x.min_dist(dest):
        #     #                 x._min_dist[dest] = xy_dist
        #
        # if dest in self._min_dist:
        #     return self._min_dist[dest]

    def __repr__(self):
        return f'Node {self.name}'

    # def init_min_dist_table(self):
    #     for dest_node in Earth.nodes:
    #         min_dist = 99999999
    #         for x in self.tracks():
    #             if x.begins_at == dest_node or x.ends_at == dest_node:  # if directly connected
    #                 if x.track_length <= min_dist:
    #                     min_dist = x.track_length
    #                     self.min_distances[dest_node] = x.track_length
    #         if min_dist != 99999999:
    #             self.min_distances[dest_node] = min_dist
    #
    #     for dest_node in Earth.nodes:
    #         if dest_node not in self.min_distances:
    #             pass


    # def min_dist(self, dest_node):
    #     if not self.min_distances:
    #
    #         for x in Earth.nodes:
    #
    #         min_dist = 99999999
    #         for x in self.tracks():
    #             if x.begins_at == dest_node or x.ends_at == dest_node:  # if directly connected
    #                 if x.track_length <= min_dist:
    #                     min_dist = x.track_length
    #                     self.min_distances[dest_node] = x.track_length
    #
    #         return min_dist
    #
    # def hop(self, dest_node, curr_dest):
    #     pass

class RoutingTable:
    def __init__(self):
        pass

    def dist(self, via, dest):
        pass

    def min_dist(self, dest):
        pass

    def next(self, node):
        pass


class Earth:  # static class
    tracks = set()
    nodes = set()
    points = set()

    def __init__(self):
        pass


if __name__ == '__main__':



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

    Earth.nodes = {na, nb, tb, tc, td, tf, tt, tw}
    Earth.tracks = {ra, rb, rc, rd, re, rf, rg, rh, ri}

    for x in Earth.nodes:
        x.nodes()  # initialize all connected nodes

    train = Train(track=rd, dest=tb, dist=0.0, direction=-1, name='bob')

    fake = Node('FAKE', Point(x=7.0, y=7.0))

    assert fake == tw  # because same location
    assert fake != tt  # because not same location

    # connected tracks
    assert tw.tracks() == {ri}
    assert td.tracks() == {ra, rb, rd}

    # connected nodes
    assert tw.nodes() == {na}
    assert td.nodes() == {tc, nb, tb}


    train.step()











    # assert tt.min_dist(tw) == 4
    # assert tt.min_dist(tf) == 4
    # # assert tt.min_dist(tb) == 8
    # # assert tt.min_dist(tc) == 8
    # # assert tt.min_dist(td) == 8
    #
    # assert tt.routing_table.next(tw) == na
    # assert tt.routing_table.next(tf) == na
    # assert tt.routing_table.next(tb) == na
    # assert tt.routing_table.next(tc) == na
    # assert tt.routing_table.next(td) == na
    #
    #
    #
    # assert nb.routing_table.min_dist(tw) == 5
    # assert nb.routing_table.min_dist(tt) == 5
    # assert nb.routing_table.min_dist(tf) == 5
    # assert nb.routing_table.min_dist(tb) == 3
    # assert nb.routing_table.min_dist(tc) == 3
    # assert nb.routing_table.min_dist(td) == 3
    #
    # assert nb.routing_table.next(tw) == na
    # assert nb.routing_table.next(tt) == na
    # assert nb.routing_table.next(tf) == na
    # assert nb.routing_table.next(tb) == tb
    # assert nb.routing_table.next(tc) == tc
    # assert nb.routing_table.next(td) == td
    #
    # assert nb.routing_table.dist(via=tb, dest=td) == 8
    # assert nb.routing_table.dist(via=tc, dest=td) == 8
    #
    # train.go_to(tw)
    #
    # train.step()
    # assert train.on_track == rd
    # assert train.distance_from_beginning_of_track == 1.0
    #
    # train.step()
    # train.step()
    # train.step()
    #
    # assert train.on_track == rf
    # assert train.distance_from_beginning_of_track == 2.0
    # train.step()
    # assert train.distance_from_beginning_of_track == 1.0
    # train.step()
    # assert train.distance_from_beginning_of_track == 0.0
    # train.step()
    # assert train.on_track == ri
    # assert train.distance_from_beginning_of_track == 1.0
    # train.step()
    # assert train.distance_from_beginning_of_track == 2.0
