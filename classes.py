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

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return self.position.__hash__()

    def tracks(self):
        return {x for x in Earth.tracks if x.begins_at == self or x.ends_at == self}


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
    na = Node('A', Point(x=0.0, y=0.0))
    nb = Node('B', Point(x=1.0, y=1.0))

    tb = Node('Bravo', Point(x=2.0, y=2.0))
    tc = Node('Charlie', Point(x=3.0, y=3.0))
    td = Node('Dingo', Point(x=4.0, y=4.0))
    tf = Node('Foxtrot', Point(x=5.0, y=5.0))
    tt = Node('Tango', Point(x=6.0, y=6.0))
    tw = Node('Whiskey', Point(x=7.0, y=7.0))

    ra = Track(tc, td, 5)
    rb = Track(td, tb, 5)

    rc = Track(tb, nb, 3)
    rd = Track(td, nb, 3)
    re = Track(tc, nb, 3)

    rf = Track(na, nb, 3)

    rg = Track(na, tf, 2)
    rh = Track(na, tt, 2)
    ri = Track(na, tw, 2)

    Earth.nodes |= {na, nb, tb, tc, td, tf, tt, tw}
    Earth.tracks |= {ra, rb, rc, rd, re, rf, rg, rh, ri}


    train = Train(track=rd, dest=tb, dist=0.0, direction=-1, name='bob')

    fake = Node('FAKE', Point(x=7.0, y=7.0))

    assert fake == tw  # because same location
    assert fake != tt  # because not same location

    assert tw.tracks() == {ri}
    assert td.tracks() == {ra, rb, rd}

    assert tt.routing_table.min_dist(tw) == 4
    assert tt.routing_table.min_dist(tf) == 4
    assert tt.routing_table.min_dist(tb) == 8
    assert tt.routing_table.min_dist(tc) == 8
    assert tt.routing_table.min_dist(td) == 8

    assert tt.routing_table.next(tw) == na
    assert tt.routing_table.next(tf) == na
    assert tt.routing_table.next(tb) == na
    assert tt.routing_table.next(tc) == na
    assert tt.routing_table.next(td) == na



    assert nb.routing_table.min_dist(tw) == 5
    assert nb.routing_table.min_dist(tt) == 5
    assert nb.routing_table.min_dist(tf) == 5
    assert nb.routing_table.min_dist(tb) == 3
    assert nb.routing_table.min_dist(tc) == 3
    assert nb.routing_table.min_dist(td) == 3

    assert nb.routing_table.next(tw) == na
    assert nb.routing_table.next(tt) == na
    assert nb.routing_table.next(tf) == na
    assert nb.routing_table.next(tb) == tb
    assert nb.routing_table.next(tc) == tc
    assert nb.routing_table.next(td) == td

    assert nb.routing_table.dist(via=tb, dest=td) == 8
    assert nb.routing_table.dist(via=tc, dest=td) == 8

    train.go_to(tw)

    train.step()
    assert train.on_track == rd
    assert train.distance_from_beginning_of_track == 1.0

    train.step()
    train.step()
    train.step()

    assert train.on_track == rf
    assert train.distance_from_beginning_of_track == 2.0
    train.step()
    assert train.distance_from_beginning_of_track == 1.0
    train.step()
    assert train.distance_from_beginning_of_track == 0.0
    train.step()
    assert train.on_track == ri
    assert train.distance_from_beginning_of_track == 1.0
    train.step()
    assert train.distance_from_beginning_of_track == 2.0
