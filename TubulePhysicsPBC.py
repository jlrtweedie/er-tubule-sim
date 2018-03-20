import random
import numpy as np


class Junction:
    """ Intersections and connections between tubules/boundary """

    def __init__(self, x, y):

        self.position = np.array([x, y])
        self.velocity = np.array([0.0, 0.0])
        self.vector = np.array([0, 0])

        self.flowing = False
        self.growing = False
        self.anchor = False
        self.flowrate = np.array([0.0, 0.0])
        self.dragrate = 1
        self.cross = 0
        self.width = 0
        self.adjacent = []
        self.just_split = True

    def moveJunction(self):
        """ Updates position with given velocity, acceleration,
            and drag """

        # Wraparound happens at the junction level - buggy to do
        # on the tubules directly
        if not self.anchor:
            if self.cross > 0:
                self.position[0] %= self.width
                self.cross = 0
            if self.cross < 0:
                self.position[0] %= self.width
                self.cross = 0

        # System flow still unimplemented
        if self.flowing:
            self.position += self.flowrate

        if not self.anchor:
            self.position += self.velocity
            self.velocity *= self.dragrate


class Tubule:
    """ Tubules defined by two junction endpoints """

    def __init__(self, j1, j2):

        self.j1 = j1
        self.j2 = j2
        self.width = 0
        self.crossover = 0

        # Most the tubule physics are done with vectors instead
        # of absolute positions
        self.vector = self.j2.position - self.j1.position
        self.vector[0] += self.width * self.crossover

        self.norm = np.linalg.norm(self.vector)
        self.unit = self.vector / np.linalg.norm(self.vector)

        self.hookes = 0
        self.growth = 0
        self.contracting = True

    def updateTubule(self):
        """ Updates self.vector once per unit time """

        self.vector -= self.vector
        self.vector += self.j2.position
        self.vector[0] += self.width * self.crossover
        self.vector -= self.j1.position
        self.norm = np.linalg.norm(self.vector)
        self.unit = self.vector / np.linalg.norm(self.vector)

    def contractTubule(self):
        """ Pulls tubule ends together (Hooke's Law) """

        if not self.j1.anchor and self.contracting:
            self.j1.velocity += self.hookes * self.vector
        if not self.j2.anchor and self.contracting:
            self.j2.velocity -= self.hookes * self.vector

    def growTubule(self):
        """ Grows open tubules """

        if self.j1.growing:
            self.j1.velocity -= self.hookes * self.vector
            self.j1.position -= self.growth * self.vector / self.norm
        if self.j2.growing:
            self.j2.velocity += self.hookes * self.vector
            self.j2.position += self.growth * self.vector / self.norm


class Substrate:
    """ Defines simulation boundaries and properties """

    def __init__(self, width, height, **kargs):

        self.width = width
        self.height = height

        # Lots of kargs to pass default values to junctions/tubules
        self.bound_x = kargs.get('bound_x', False)
        self.wrap_x = kargs.get('wrap_x', True)

        self.flowing = kargs.get('flowing', False)
        self.flowrate = kargs.get('flowrate', np.array([0.0, -0.0000001]))
        self.dragrate = kargs.get('dragrate', 0.99)
        self.hookes = kargs.get('hookes', 0.00005)
        self.growth = kargs.get('growth', 0.5)
        self.contracting = kargs.get('contracting', True)

        self.junctions = []
        self.tubules = []
        self.j_functions = []
        self.t_functions = []

        # Dictionary of anonymous functions - can enable/disable when
        # calling the substrate at the beginning of the simulation script
        self.function_dict = {
            'moveJunction': (1, lambda j: j.moveJunction()),
            'boundSubstrate': (1, lambda j: self.boundSubstrate(j)),
            'mergeTubule': (1, lambda j: self.mergeTubule(j)),
            'updateTubule': (2, lambda t: t.updateTubule()),
            'contractTubule': (2, lambda t: t.contractTubule()),
            'growTubule': (2, lambda t: t.growTubule()),
            'pruneTubule': (2, lambda t: self.pruneTubule(t)),
            'wrapSubstrate': (2, lambda t: self.wrapSubstrate(t))
        }

    def addFunctions(self, function_list):
        """ Adds anonymous functions to run each timestep """

        for func in function_list:
            (n, f) = self.function_dict.get(func, (-1, None))
            if n == 1:
                self.j_functions.append(f)
            elif n == 2:
                self.t_functions.append(f)
            else:
                print("No such function: %s" % func)

    def addJunction(self, x, y, **kargs):
        """ Creates a junction """

        junction = Junction(x, y)

        junction.anchor = kargs.get('anchor', False)
        junction.growing = kargs.get('growing', False)
        junction.flowing = kargs.get('flowing', self.flowing)
        junction.flowrate = kargs.get('flowrate', self.flowrate)
        junction.dragrate = kargs.get('dragrate', self.dragrate)
        junction.width = kargs.get('width', self.width)
        junction.cross = kargs.get('cross', 0)

        self.junctions.append(junction)
        return(junction)

    def addTubule(self, j1, j2, **kargs):
        """ Creates a tubule between two junctions """

        tubule = Tubule(j1, j2)

        tubule.hookes = kargs.get('hookes', self.hookes)
        tubule.growth = kargs.get('growth', self.growth)
        tubule.width = kargs.get('growth', self.width)
        tubule.crossover = kargs.get('crossover', 0)

        j1.adjacent.append(j2)
        j2.adjacent.append(j1)

        self.tubules.append(tubule)
        return(tubule)

    def splitTubule(self):
        """ Picks a point along the system length to grow
            a new tubule from """

        # Sum vector length and pick a random float less than that sum
        sys_len = sum(t.norm for t in self.tubules)
        split_len = random.uniform(0, sys_len)

        for t in self.tubules:

            # This method picks which tubule to split from by continually
            # subtracting the norm of each vector from the system length
            # until the remaining length is smaller than the norm of the
            # current vector
            if split_len > t.norm:
                split_len -= t.norm

            else:

                s2_vec = np.array([-t.unit[1], t.unit[0]])
                s2_vec *= random.choice([-1, 1])

                s1_pos = t.j1.position + split_len * t.unit
                s2_pos = s1_pos + 0.1 * s2_vec

                s_j1_crossover = 0
                s_j2_crossover = 0
                sp_crossover = 0

                # This stuff is kind of funky but prevents crossover
                # errors
                if t.crossover > 0:
                    if s1_pos[0] < t.j2.position[0]:
                        s_j1_crossover = -1
                    elif s1_pos[0] > t.j1.position[0]:
                        s_j2_crossover = 1
                elif t.crossover < 0:
                    if s1_pos[0] < t.j1.position[0]:
                        s_j2_crossover = -1
                    elif s1_pos[0] > t.j2.position[0]:
                        s_j1_crossover = 1

                if s2_pos[0] > self.width and s1_pos[0] < self.width:
                    sp_crossover += 1
                elif s2_pos[0] < 0 and s1_pos[0] > 0:
                    sp_crossover -= 1

                s1 = self.addJunction(s1_pos[0], s1_pos[1])
                s2 = self.addJunction(s2_pos[0], s2_pos[1], growing=True)

                # Not sure why but removing junctions from another
                # junction's adjacency list sometimes raises exceptions
                # in this method but not in mergeTubule
                try:
                    t.j1.adjacent.remove(t.j2)
                except ValueError:
                    pass
                try:
                    t.j2.adjacent.remove(t.j1)
                except ValueError:
                    pass

                self.addTubule(s1, t.j1, crossover=s_j1_crossover)
                self.addTubule(s1, t.j2, crossover=s_j2_crossover)
                self.addTubule(s1, s2, crossover=sp_crossover)

                t.to_remove = t

                return

    def mergeTubule(self, j):
        """ Connects tubules when the open end of one tubule
            intersects a different tubule it isn't already
            attached to """

        if j.growing:

            # Prevents self attachment
            if j.just_split:
                j.just_split = False
                return

            # This is a nested loop that runs pretty inefficiently -
            # this method is where you should look if you want to
            # optimize this sim
            for t in self.tubules:

                if j in t.j1.adjacent:
                    pass
                elif j in t.j2.adjacent:
                    pass

                else:

                    # I'd prefer not to assign all these variables
                    # each time this function loops around but
                    # I'd need to figure out a better way to check
                    # all the vector values quickly
                    j.vector = j.position - t.j1.position
                    if t.crossover > 0:
                        if j.position[0] < t.j2.position[0]:
                            j.vector[0] += self.width
                    elif t.crossover < 0:
                        if j.position[0] > t.j2.position[0]:
                            j.vector[0] -= self.width
                    j_norm = np.linalg.norm(j.vector)
                    j_unit = j.vector / j_norm

                    # Sets conditions for merging - more
                    # forgiving when j_norm is small to prevent
                    # crossovers
                    if j_norm < 0.1:
                        col_check = 0
                    elif j_norm < 1:
                        col_check = 0.9
                    else:
                        col_check = 0.999

                    # Prevents attachment from outside the tubule's
                    # max length
                    if j_norm > t.norm:
                        pass

                    # This is the most accurate algorithm I've been
                    # able to come up with - it dots the vector from
                    # tubule end 1 to tubule end 2 with the vector
                    # from tubule end 1 to the open junction, and if
                    # they're close enough to collinear then it attaches
                    elif np.dot(j_unit, t.unit) > col_check:

                        j1_crossover = 0
                        j2_crossover = 0

                        # Prevents crossover errors
                        if t.crossover > 0:
                            if j.position[0] < t.j2.position[0]:
                                j1_crossover = -1
                            elif j.position[0] > t.j1.position[0]:
                                j2_crossover = 1
                        elif t.crossover < 0:
                            if j.position[0] < t.j1.position[0]:
                                j2_crossover = -1
                            elif j.position[0] > t.j2.position[0]:
                                j1_crossover = 1

                        self.addTubule(j, t.j1, crossover=j1_crossover)
                        self.addTubule(j, t.j2, crossover=j2_crossover)

                        j.adjacent.remove(t.j1)
                        j.adjacent.remove(t.j2)

                        j.growing = False
                        t.to_remove = t
                        break

    def pruneTubule(self, t):
        """ Removes tubules given certain conditions - unimplemented"""

    def pruneJunction(self, j):
        """ Removes junctions given certain conditions - unimplemented """

    def boundSubstrate(self, j):
        """ Anchors junctions when they hit a boundary - y bounds
            are enabled by default and x bounds can be enabled for
            debugging """

        if self.bound_x:
            if j.position[0] > self.width:
                j.position[0] = self.width
                j.anchor = True
                j.growing = False
            elif j.position[0] < 0:
                j.position[0] = 0
                j.anchor = True
                j.growing = False

        if j.position[1] > self.height:
            j.position[1] = self.height
            j.anchor = True
            j.growing = False
        elif j.position[1] < 0:
            j.position[1] = 0
            j.anchor = True
            j.growing = False

    def wrapSubstrate(self, t):
        """ Keeps track of tubules that cross the x bound and which
            junction crossed - works without need for a new anchor
            point attached to the side walls """

        if self.wrap_x:

            if t.j1.position[0] > self.width:
                t.crossover -= 1
                if t.crossover < -1:
                    t.crossover = -1
                t.j1.cross += 1
                if t.j1.cross > 1:
                    t.j1.cross = 1
            elif t.j1.position[0] < 0:
                t.crossover += 1
                if t.crossover > 1:
                    t.crossover = 1
                t.j1.cross -= 1
                if t.j1.cross < -1:
                    t.j1.cross = -1

            if t.j2.position[0] > self.width:
                t.crossover += 1
                if t.crossover > 1:
                    t.crossover = 1
                t.j2.cross += 1
                if t.j2.cross > 1:
                    t.j2.cross = 1
            elif t.j2.position[0] < 0:
                t.crossover -= 1
                if t.crossover < -1:
                    t.crossover = -1
                t.j2.cross -= 1
                if t.j2.cross < -1:
                    t.j2.cross = -1

    def initSubstrate(self):
        """ Creates two initial vertical tubules """

        x_bot = sorted([random.uniform(0, self.width),
                        random.uniform(0, self.width)])
        x_top = sorted([random.uniform(0, self.width),
                        random.uniform(0, self.width)])

        jbot_1 = self.addJunction(x_bot[0], 0, anchor=True)
        jbot_2 = self.addJunction(x_bot[1], 0, anchor=True)
        jtop_1 = self.addJunction(x_top[0], self.height, anchor=True)
        jtop_2 = self.addJunction(x_top[1], self.height, anchor=True)

        self.addTubule(jbot_1, jtop_1)
        self.addTubule(jbot_2, jtop_2)

    def updateSubstrate(self):
        """ Runs all functions that need to run each timestep """

        for f in self.j_functions:
            for junction in self.junctions:
                f(junction)

        for f in self.t_functions:
            for tubule in self.tubules:
                f(tubule)
