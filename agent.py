
import numpy as np
from scipy.signal import correlate2d as cor

class Agent:

    NOISE_SENSE = 0.1
    NOISE_MOVEMENT = {-1:0., 0:1., 1:0.}

    def __init__(self, environment):

        self.env = environment
        self.p = np.zeros((self.env.M, self.env.N))
        self.p.fill(1/(self.env.N*self.env.M))


    def __str__(self):

        return "{}".format(np.round(self.p, 3))


    def sense(self, obs):

        q = self.p[:,:]
        for i in range(self.env.M):
            for j in range(self.env.N):
                if np.all(self.env.array_world[:, i, j] == obs):
                    q[i, j] = self.p[i, j]*(1-self.NOISE_SENSE)
                else:
                    q[i, j] = self.p[i, j]*  self.NOISE_SENSE

        self.p = q / q.sum()


    def do_move(self, move=None):
        # if move is None select random move
        if move == None:
            move = (np.random.randint(-1, 2), np.random.randint(-1, 2))

        # update beilefs after move
        filter = self._get_filter_for_movement(move)
        self.p = cor(self.p, filter, "same", "wrap")

        return move


    def _get_filter_for_movement(self, move):
        # make filter for move[0] and move[1] baseed on i, j
        filter_0 = np.array([[self.NOISE_MOVEMENT[1] for i in range(len(self.NOISE_MOVEMENT))],
                             [self.NOISE_MOVEMENT[0] for i in range(len(self.NOISE_MOVEMENT))],
                             [self.NOISE_MOVEMENT[-1] for i in range(len(self.NOISE_MOVEMENT))]])
        filter_1 = filter_0.T

        # shift by -move
        filter_0 = np.roll(filter_0, -move[0], axis=0)
        filter_1 = np.roll(filter_1, -move[1], axis=1)

        return filter_0*filter_1


    def MAP_estimate_location(self):

        return np.unravel_index(self.p.argmax(), self.p.shape)
