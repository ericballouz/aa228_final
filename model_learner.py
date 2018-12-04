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
        self.alpha = 0.9
        self.gamma = 0.9
        self.dt = 0.1

    def learn(self):
        # Q unlikely to converge so update set number of times
        curr_state = get_start_state()
        episode_count = 0
        action_count = 0
        while episode_count < 100 or action_count < 10000:
            # choose action a based on exploration strategy
            curr_action = model.nextAction(curr_state, self.Q, self.N)
            # observe reward r_t
            reward_t, game_finished = model.R(curr_state)

            self.N[curr_state + curr_action] += 1
            Q_t = self.Q[curr_state + curr_action]

            # tuple([x, y, V, th]) = nextState(a)
            # observe new state s_t+1
            next_state = model.nextState(curr_state, curr_action, self.dt)
            delta = reward_t + (self.gamma*max([self.Q[next_state + action] for action in model.action_space()])) - Q_t

            # propagate rewards
            for key in self.N.keys():
                self.Q[key] += self.alpha * delta * self.N[key]
                self.N[key] += self.gamma * self.lam * self.N[key]

            action_count += 1
            # reset when simulation ends
            if game_finished:
                episode_count += 1
                self.N.clear()
                curr_state = get_start_state()
            else:
                curr_state = next_state

        print("Done learning with episode count:{}  and action count: {}".format(episode_count, action_count))

    # input: state and action
    # output: closest match within self.Q
    def closestStateAction(self, state, action):
        state_action = state + action
        min_distance = 10000
        closest = 0
        for learned_state in self.Q.keys():
            distance = sum([(state_action[i] - learned_state[i])**2] for i in range(len(state_action)))
            if distance <= min_distance:
                min_distance = distance
                closest = learned_state
            # if learned_state close enough, return
            if distance <= 3.02: return learned_state
        return closest

    def best_action(self, state):
        """
        Return the best action from a trained Q model for a given state
        :param: state tuple (x, y, V, th)
        :return: action tuple (dV, dth)
        """
        action_space = model.action_space()
        max_idx = np.argmax([self.Q[closestStateAction(state, action)] for action in action_space])
        return action_space[max_idx]

if __name__ == '__main__':
    print("Running model_learner.py main")
    my_model = Qlearn()
