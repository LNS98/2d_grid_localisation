
from environment import Environment
from agent import Agent

import numpy as np

# np.random.seed(2)

def main():

    env = Environment(10, 10, magnification=80)
    robot = Agent(env)
    env.add_agent(robot)
    env.starting_agent_location()

    for i in range(50):

        # make the robot sense the env and update beiliefs
        robot.sense(env.get_obs())
        env.draw()
        env.step(robot.do_move())

    # print(env.agent_state)
    # print(robot)
    # print(robot.p.sum())

    return None



if __name__ == "__main__":

    main()
