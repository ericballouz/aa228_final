import numpy as np
import random
import itertools as it

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
        r += 70
        return r, True

    return r, False

# input: state, action, time step
# output: next state
def nextState(s, a, dt):
    x, y, V, theta = (s[0], s[1], s[2], s[3])
    dV, dtheta = (a[0], a[1])
    x = x+V*np.sin(theta)*dt
    y = y+V*np.cos(theta)*dt
    theta += dtheta
    V += dV*dt
    return_val = adjustValues(x, y, V, theta)
    assert len(return_val) == 4
    return return_val

def adjustValues(x, y, V, theta):
    x = np.round(x/5)*5
    y = np.round(y/5)*5
    V = max(0, V)
    V = np.round(V)
    theta = np.mod(theta, np.pi)
    return (x, y, V, theta)

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


def BoltzmannExplore(s, Q_dict, trackCompleted):
    """
        implements Boltzmann exploration
        s: current state
        Q_dict: dictionary of Q values involving s
        returns an action
    """
    # calculate probabilities
    tau = 0.5
    if trackCompleted: tau = 0.65
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

def epsGreedy(s, Q_dict):
    epsilon = 0.1
    A = action_space()
    imax = np.argmax([Q_dict[(s+a)] for a in A])
    U = random.uniform(0, 1)
    if U < (1-epsilon): return A[imax]
    else:
        i = random.randint(0, len(A)-1)
        return A[i]

def nextAction(s, Q_dict, trackCompleted):
    return BoltzmannExplore(s, Q_dict, trackCompleted)#epsGreedy(s, Q_dict)

def get_state_space(world):
    x_space = np.zeros(int(world.window_w/5)+1)
    counter = 0
    for i in range(int(-world.window_w/2),int(world.window_w/2),5):
        x_space[counter]=i
        counter+=1
    x_space[x_space.size-1] = int(world.window_w/2)

    y_space = np.zeros(int(world.window_l/5)+1)
    counter = 0
    for i in range(int(-world.window_l/2),int(world.window_l/2),5):
        y_space[counter] = i
        counter+=1
    y_space[y_space.size-1] = int(world.window_l/2)

    V_space = np.zeros(101)
    for i in range(101):
        V_space[i] = i

    theta_space = (-np.pi)*np.ones(17)
    for i in range(theta_space.size):
        theta_space[i]+=(np.pi/8)*i
    #theta_space = np.around(theta_space,1)
    print(theta_space)

    return x_space,y_space,V_space,theta_space

def get_state_space_dict(world):
    x_space, y_space, V_space, theta_space = get_state_space(world)
    state_space_dict = dict()
    space_list = [list(x_space),list(y_space), list(V_space),list(theta_space)]
    for element in it.product(*space_list):
        state_space_dict[element] = 0
        #for xi in range(x_space.size):
            #for yi in range(y_space.size):
                #for Vi in range(V_space.size):
                    #for thetai in range(theta_space.size):
                        #state_space_dict[(xi,yi,Vi,thetai)] = 0
    return theta_space,state_space_dict

def find_closest_theta(theta,theta_space):
    diff_list = np.zeros(theta_space.size)
    for theta_test_ind in range(theta_space.size):
        diff = np.abs(theta_space[theta_test_ind]-theta)
        diff_list[theta_test_ind] = diff
    return theta_space[np.argmax(diff_list)]

