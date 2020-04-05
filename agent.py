
import numpy as np

class Agent:

    SUROUNDING_CELLS = 8
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
        q = np.zeros((self.env.M, self.env.N))
        for i in range(self.env.M):
            for j in range(self.env.N):
                    # uncertainty in movement
                    q[i,j] = self._add_noise_in_movement(move, i, j)
        self.p = q

        return move


    def _add_noise_in_movement(self, move, i, j):

        # make filter for move[0] and move[1] baseed on i, j
        filter_0 = np.array([[self.NOISE_MOVEMENT[1] for i in range(len(self.NOISE_MOVEMENT))],
                             [self.NOISE_MOVEMENT[0] for i in range(len(self.NOISE_MOVEMENT))],
                             [self.NOISE_MOVEMENT[-1] for i in range(len(self.NOISE_MOVEMENT))]])
        filter_1 = filter_0.T

        # shift by -move
        filter_0 = np.roll(filter_0, -move[0], axis=0)
        filter_1 = np.roll(filter_1, -move[1], axis=1)

        # select subset of p:p[i-1:i+1, j-1:j+1]
        # p_i_inds = [(i-1)%self.env.M, i, (i+1)%self.env.M]
        # p_j_inds = [(j-1)%self.env.N, j, (j+1)%self.env.N]

        p_ij = self._get_p_subset(i, j)

        # print(p_ij)
        # print("P_ij")

        # convolve the filters
        result = (p_ij*filter_0*filter_1).sum()

        return result

    def _get_p_subset(self,i, j):

        shifted_p = self.p[:]

        i_roll, j_roll = 0, 0
        if i == 0:
            i_roll = 1
        if j == 0:
            j_roll = 1
        if i == self.env.M-1:
            i_roll = -1
        if j == self.env.N-1:
            j_roll = -1



        # print("i_roll", i_roll, "j_roll", j_roll)
        # print("i", i, "j", j)

        shifted_p = np.roll(shifted_p, i_roll, axis=0)
        shifted_p = np.roll(shifted_p, j_roll, axis=1)

        # print("Shifted_p")
        # print(shifted_p)

        return shifted_p[i+i_roll-1:i+i_roll+2, j+j_roll-1:j+j_roll+2]




    def MAP_estimate_location(self):

        return np.unravel_index(self.p.argmax(), self.p.shape)
