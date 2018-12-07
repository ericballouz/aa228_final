from model_learner import Qlearn
import model
from world_master import World
import numpy as np

world = World()

qlearn = Qlearn()
dt = 1



theta_space, state_space_dict = qlearn.valueIterate()
#k = 0
niters  = 1
A_space = model.action_space()
state_space = state_space_dict.keys()
print(len(state_space))
print("starting loop")
for i in range(niters):
    for state in state_space:
        Rsa,_ = model.R(world,state)
        #a_dict = dict() #key:a, value:in_brackets
        max_in_brackets = -1000000 #really small
        for a in A_space:
            x_prime,y_prime,v_prime,theta_prime = model.nextState(state,a,dt)
            x_prime_adj, y_prime_adj, v_prime_adj, theta_prime_adj = model.adjustValues(x_prime,y_prime,v_prime,theta_prime)
            s_prime = x_prime_adj, y_prime_adj, v_prime_adj, model.find_closest_theta(theta_prime_adj,theta_space)
            #print('s is {}, a is {}, and sprime is {}'.format(state, a, s_prime))
            if s_prime in state_space:
                U_s_prime = state_space_dict[s_prime]
                in_brackets = Rsa + U_s_prime
                if in_brackets>max_in_brackets:
                    max_in_brackets=in_brackets
        #loop through all a to find maximum
        #in_brackets_list = list(a_dict.values())
        state_space_dict[state] = max_in_brackets
        #a_list = list(a_dict.keys())
print(state_space_dict)


