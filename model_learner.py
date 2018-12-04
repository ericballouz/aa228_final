import numpy as np
from collections import defaultdict
from start_state import get_start_state
import model


class Qlearn(object):
    def __init__(self):
        print(len(model.action_space()))
        # state_size = x_size * y_size * v_size * theta_size
        # action_size = len(model.action_space())
        self.Q = defaultdict(int)  # key: state,action(x, y, th, dv, dth) => value: Q
        self.N = defaultdict(int)  # key: (state, action) => value: N, num of visit to (state, action) for curr episode
        self.lam = 0.9
        self.alpha = 1.0
        self.gamma = 0.9
        self.tau = 0.5
        self.dt = 0.1

    def learn(self):
        # Q unlikely to converge so update set number of times
        curr_state = get_start_state()
        for _ in range(10000):
            # choose action a
            curr_action = model.nextAction(curr_state, self.Q, self.N)
            reward_t = R(curr_state)
            self.N[(curr_state, curr_action)] += 1
            Q_t = self.Q[(curr_state, curr_action)]

            # x, y, V, th = nextState(a)
            next_state = model.nextState(curr_state, curr_action, self.dt)
            delta = reward_t + self.gamma*( max([ self.Q[(next_state, action)] for action in model.action_space()]) - Q_t)
            
            #propagate rewards

            #reset when simulation ends


if __name__ == '__main__':
    print("Running model_learner.py main")
    my_model = Qlearn()
