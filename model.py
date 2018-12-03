import numpy as np
import check_if_on_track


# input: state, and number of steps so far
# output: reward of the state and whether the simulation continues
# the simulation stops when we exit the world
def R(s, N):
    if check_if_on_track.check_if_car_in_world(s):
        return -1000000, False

    r = -0.1
    if check_if_on_track.check_if_car_on_track(s):
        r += 1000/N
    return r, True

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
