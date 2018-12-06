import numpy as np
import random

# input: state, and number of steps so far
# output: reward of the state, game done or not
def R(world,s):
    if not world.check_if_car_in_world(s):
        return -10, True

    if not world.check_if_car_on_track(s):
        return -0.01, True #-10, True

    r = -0.0001
    if world.update_checkpts_seen(s):
        r += 100/world.num_checkpts

    if world.successfully_finished(s):
        r += 50
        return r, True

    return r, False

# input: state, action, time step
# output: next state
def nextState(s, a, dt):
    x, y, V, theta = (s[0], s[1], s[2], s[3])
    dV, dtheta = (a[0], a[1])
    x = x+V*np.cos(theta)*dt
    y = y+V*np.sin(theta)*dt
    theta += dtheta
    V += dV*dt
    # can't go in reverse
    if V < 0:
        V = 0
    return_val = tuple([x, y, V, theta])
    assert len(return_val) == 4
    return tuple([x, y, V, theta])

# returns action space
def action_space():
    """
    Returns all possible actions that can be taken
    :return: list of action tuples - [(dV_1, dth_1), (dV_2, dth_2), (dV_3, dth_3), ...]
    """
    dV_min, dV_max, dV_step = (-10, 10, 1.0)
    dtheta_min, dtheta_max, dtheta_step = (-np.pi/8.0, np.pi/8.0, np.pi/8.0)
    
    possible_actions = []
    for dV in np.arange(dV_min, dV_max + dV_step, dV_step):
        possible_actions.append(tuple([dV, 0]))
    for dtheta in [dtheta_min, dtheta_max]:
        for dV in np.arange(dV_min, dV_max + dV_step, dV_step):
            possible_actions.append(tuple([dV, dtheta]))

    return possible_actions


def BoltzmannExplore(s, Q_dict):
    """
        implements Boltzmann exploration
        s: current state
        Q_dict: dictionary of Q values involving s
        returns an action
    """
    # calculate probabilities
    tau = 0.505
    A = action_space()
    p = {}
    normalize = 0
    for a in A:
        p[a] = np.exp(Q_dict[s + a]/tau)
        normalize += p[a]

    # sample according to acceptance-rejection
    i = random.randint(0, len(A)-1)
    Z = A[i]
    U = random.uniform(0, 1)
    while p[Z]/normalize <= U:
        i = random.randint(0, len(A)-1)
        Z = A[i]
        U = random.uniform(0, 1)     

    assert len(Z) == 2
    return Z

def nextAction(s, Q_dict):
    return BoltzmannExplore(s, Q_dict)

