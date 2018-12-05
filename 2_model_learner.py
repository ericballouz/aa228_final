import numpy as np
from collections import defaultdict
#from start_state import get_start_state
import model
from world_master import World
import plot_world_v2

class Qlearn(object):
    def __init__(self):
        print(len(model.action_space()))
        # state_size = x_size * y_size * v_size * theta_size
        # action_size = len(model.action_space())
        self.Q = defaultdict(int)  # key: state,action(x, y, th, dv, dth) => value: Q
        self.N = defaultdict(int)  # key: (state, action) => value: N, num of visit to (state, action) for curr episode
        self.lam = 0.8
        self.alpha = 0.9
        self.gamma = 0.9
        self.dt = 0.1
        self.world = World()
        #self.minimum_Q = 0
        self.seen_states = {}
        self.x_opt = []
        self.y_opt = []

    def learn(self):
        # Q unlikely to converge so update set number of times
        curr_state = self.world.get_start_state()
        x_list = [curr_state[0]]
        y_list = [curr_state[1]]

        episode_count = 0
        action_count = 0
        while episode_count < 1000:

            # choose action a based on exploration strategy
            curr_action = model.nextAction(curr_state, self.Q)
            if curr_state not in self.seen_states: self.seen_states[curr_state] = set()
            self.seen_states[curr_state].add(curr_action)
            # observe reward r_t
            reward_t, game_finished = model.R(self.world, curr_state)

            self.N[curr_state + curr_action] += 1
            Q_t = self.Q[curr_state + curr_action]
            #if Q_t <= self.minimum_Q: self.minimum_Q = Q_t

            # tuple([x, y, V, th]) = nextState(a)
            # observe new state s_t+1
            next_state = model.nextState(curr_state, curr_action, self.dt)
            x_list.append(next_state[0])
            y_list.append(next_state[1])
            delta = reward_t + (self.gamma*max([self.Q[next_state + action] for action in model.action_space()])) - Q_t

            # propagate rewards
            for key in self.N.keys():
                self.Q[key] += self.alpha * delta * self.N[key]
                self.N[key] = self.gamma * self.lam * self.N[key]

            action_count += 1
            # reset when simulation ends
            if game_finished or action_count > 5000:
                episode_count += 1
                self.N.clear()
                curr_state = self.world.get_start_state()
                print("Done learning with episode count:{}  and action count: {}".format(episode_count, action_count))
                #if episode_count % 50 == 0:
                #    plot_world_v2.plot_startup(self.world)
                #    plot_world_v2.plot_trajectory(np.asarray(x_list), np.asarray(y_list))
                #    plot_world_v2.show_plot()
                action_count = 0
                x_list = [self.world.start_state[0]]
                y_list = [self.world.start_state[1]]
                self.world.checkpts_hit = [0]
            else:
                curr_state = next_state



    # input: state and action
    # output: closest match within self.Q
    def closestState(self, state, action):
        min_distance = 10000
        closest = 0
        for sa in self.Q.keys():
            learned_state = (sa[0], s[1], s[2], s[3])
            distance = sum([(state[i] - learned_state[i])**2 for i in range(len(state))])
            if distance <= min_distance:
                min_distance = distance
                closest = learned_state
            # if learned_state close enough, return
            if distance <= 5.0: return learned_state #3.02
        return closest

    def best_action(self, state):
        """
        Return the best action from a trained Q model for a given state
        :param: state tuple (x, y, V, th)
        :return: action tuple (dV, dth)
        """
        action_space = self.seen_states[state]
        #max_idx = np.argmax([self.Q[self.closestState(state)+ action] for action in action_space])
        Qmax = -1000000
        amax = (0, 0)
        for action in action_space:
            if self.Q[state + action] >= Qmax:
                Qmax = self.Q[state + action]
                amax = action
        return action_space[max_idx]

    def optimalPolicy(self):
        curr_state = self.world.get_start_state()
        policy = [curr_state]
        endGame = False 
        i = 0;
        while not endGame and i<5000:
            i += 1
            print(i)
            optimal_action = self.best_action(curr_state)
            curr_state = model.nextState(curr_state, optimal_action, self.dt)
            policy.append(optimal_action)
            self.x_opt.append(optimal_action[0])
            self.y_opt.append(optimal_action[1])
            _, endGame = model.R(self.world, curr_state)
        plot_world_v2.plot_startup(self.world)
        plot_world_v2.plot_trajectory(np.asarray(self.x_opt), np.asarray(self.y_opt))
        plot_world_v2.show_plot()
        print(self.x_opt)
        return policy


if __name__ == '__main__':
    print("Running model_learner.py main")
    my_model = Qlearn()
    print("begin learning")
    my_model.learn()
    print("done! calculate optimal policy")
    policy = my_model.optimalPolicy()
    print("done!")
