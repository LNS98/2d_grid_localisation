
import numpy as np
import cv2
import matplotlib as plt

from agent import Agent

class Environment:

    def __init__(self, M, N, magnification=300):

        self.M  = M # rows
        self.N = N # columns
        self.mag = magnification

        # define the position of the array
        self.array_world = np.zeros([3, self.M, self.N])
        self.fill_environment()

        self.agent_state = None
        self.agent = None


        self.image = np.zeros([int(self.mag * self.M), int(self.mag * self.N), 3], dtype=np.uint8)

    def fill_environment(self):
        """
        Define the environment as a random set of red and green places
        """
        states = ["red", "green"]
        colours = {"red":(0, 0, 255), "green":(0, 255, 0)}

        for i in range(self.M):
            for j in range(self.N):

                state = np.random.choice(states)
                self.array_world[:, i, j] = colours[state]

        self.array_world = self.array_world.astype("int")

    def add_agent(self, agent):

        self.agent = agent

    def starting_agent_location(self, starting_location=None):

        if starting_location == None:
            # select the starting state as random
            starting_location = (np.random.randint(0, self.M),
                                 np.random.randint(0, self.N))

        # define the location of the agent as the starting location
        self.agent_state = starting_location


    def step(self, move):

        # get the noise
        p_noise = [i for i in self.agent.NOISE_MOVEMENT.values()]
        noise_values = [i for i in self.agent.NOISE_MOVEMENT.keys()]

        eta = [np.random.choice(noise_values, p = p_noise),
               np.random.choice(noise_values, p = p_noise)]

        self.agent_state = tuple(map(lambda x, y, z, noise: (x+y+noise)%z,
                                 self.agent_state, move, [self.M, self.N], eta))


    def get_obs(self):
        """
        noise parameter specifies the probability of getting the state from a
        surrounding cell
        """
        u = np.random.rand()
        noise = self.agent.NOISE_SENSE

        if u > noise:
            return self.array_world[:, self.agent_state[0], self.agent_state[1]]
        else:
            # select at random one the surrounding cells
            while True:
                i = (self.agent_state[0] + np.random.randint(-1, 2))%self.M
                j = (self.agent_state[1] + np.random.randint(-1, 2))%self.N
                # check that the same cell isn't selected
                if (i, j) != (self.agent_state[0], self.agent_state[1]):
                    break
            return self.array_world[:, i, j]


    def draw(self):

        self.image.fill(0)

        for i in range(1, self.N+1):
            #-------- Horizontal lines -------------
            # get the starting point
            start_tuple_hor = (0, int((i/self.N)* self.M * self.mag))
            # get the end point
            end_tuple_hor = (int(self.M * self.mag), int((i/self.N)* self.M * self.mag))
            # Draw a diagonal blue line with thickness of 5 px
            self.image = cv2.line(self.image,start_tuple_hor,end_tuple_hor, (255,255,255), 2)

        for i in range(self.M+1):
            # ------------------ Vertical Lines -----------------
            # get the starting point
            start_tuple_ver = (int((i/self.N)* self.M * self.mag), 0)
            # get the end point
            end_tuple_ver = (int((i/self.N)* self.M * self.mag), int(self.M * self.mag))
            # Draw a diagonal blue line with thickness of 5 px
            self.image = cv2.line(self.image,start_tuple_ver,end_tuple_ver, (255,255,255), 2)



        # fill drawing based on array_world
        for i in range(self.M):
            for j in range(self.N):
                # get the colour from the cell of the array world
                colour = tuple(int(x) for x in self.array_world[:, i, j])
                # get the starting and ending coordinates to fill the squares
                start_coord = (int((j/self.N)*self.M*self.mag), int((i/self.M)*self.N*self.mag))
                end_coord = (int(((j+1)/self.N)*self.M*self.mag), int(((i+1)/self.M)*self.N*self.mag))

                self.image = cv2.rectangle(self.image, start_coord, end_coord,
                                           colour, -1)


        # add the agent in
        agent_centre = (int((self.agent_state[1]+1/2)*self.mag),
                        int((self.agent_state[0]+1/2)*self.mag))
        agent_radius = int(0.1*self.mag)
        agent_colour = (0, 255, 255)
        self.image = cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)

        # add in agents MAP estimate of its location
        map_state = self.agent.MAP_estimate_location()
        agent_centre = (int((map_state[1]+1/2)*self.mag),
                        int((map_state[0]+1/2)*self.mag))
        agent_radius = int(0.1*self.mag)
        agent_colour = (255, 0, 255)
        self.image = cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)

        cv2.imshow("Environment", self.image)
        c = cv2.waitKey(int(100))
