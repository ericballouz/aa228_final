import numpy as np
import check_if_on_track
import random

# input: state, and number of steps so far
# output: reward of the state
def R(s, N):
    if not check_if_on_track.check_if_car_in_world(s):
        return -10000

    r = -0.1
    if check_if_on_track.check_if_car_on_track(s):
        r += 1000/N

    #if on finish line and all checkpoints reached
        #r += 10000
    return r

#def endGame(s, start, checkpoints):
##    if not check_if_on_track.check_if_car_in_world(s):
 #       return True
 #   if

# input: state, action, time step
# output: next state
def nextState(s, a, dt):
    x, y, V, theta = (s[0], s[1], s[2], s[3])
    dV, dtheta = (a[0], a[1])
    x = x+V*np.cos(theta)*dt
    y = V*np.sin(theta)*dt
    theta += dtheta
    V += dV*dt
    return x, y, V, theta

# returns action space
def action_space():
    dV_min, dV_max, dV_step = (-10, 10, 1.0)
    dtheta_min, dtheta_max, dtheta_step = (-np.pi/8.0, np.pi/8.0, np.pi/8.0)
    
    possible_actions = []
    for dV in np.arange(dV_min, dV_max + dV_step, dV_step):
        possible_actions.append([dV, 0])
    for dtheta in [dtheta_min, dtheta_max]:
        for dV in np.arange(dV_min, dV_max + dV_step, dV_step):
            possible_actions.append([dV, dtheta])

    return possible_actions

# implements Boltzmann exploration
# s: current state
# Q_dict: dictionary of Q values involving s
# N: number of times s was visited
# returns an action
def BoltzmannExplore(s, Q_dict, N):
    # calculate probabilities
    C = 10000
    A = action_space()
    p = {}
    beta = np.log(N)/C
    normalize = 0
    for a in A:
        p[a] = np.exp(b*Q_dict[(s, a)])
        normalize += np.exp(p[a])

    # sample according to acceptance-rejection
    i = random.uniform(0, len(A)-1)
    Z = A[i]
    U = random.uniform(0, 1)
    while p[Z]/normalize < U:
        i = random.uniform(0, len(A)-1)
        Z = A[i]
        U = random.uniform(0, 1)     
    
    return Z

def nextAction(s, Q_dict, N):
    return BoltzmannExplore(s, Q_dict, N)











