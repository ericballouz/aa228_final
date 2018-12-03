import numpy as np
import create_track


# input: state, and number of steps so far
# output: reward of the state and whether the simulation continues
# the simulation stops when we exit the world
def R(s, N):
    #

    #if outside of world
        return -1000000, False
    r = -0.1
    #if on racetrack
        r += 1000/N
    return r, True

# input: state, action, time step
# output: next state
def nextState(s, a, dt):
    x, y, V, theta = (s[0], s[1], s[2], s[3])
    dV, dtheta = (a[0], a[1])
    x = x+V*np.cos(theta)*dt
    y = V*np.sin(theta)*dt
    theta += dtheta*dt
    V += dV*dt
    return x, y, V, theta


