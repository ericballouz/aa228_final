from create_track import World
import numpy as np


def check_if_on_circle(world,x,y):
    loc = x*x + y*y
    if loc>=(world.r_i*world.r_i) and loc<=(world.r_o*world.r_o):
        return True
    return False

def check_if_on_straight(world,x,y):
    if np.abs(x)>=world.r_i and np.abs(y)<=world.track_len/2:
        return True
    return False

def check_if_car_in_world(state):
    (x, y, V, theta) = state
    world = World()
    if np.abs(x)<=world.window_w/2 and np.abs(y)<=world.window_l/2:
        return True
    return False

def check_if_car_on_track(state):
    (x,y,V,theta) = state
    world = World()
    if np.abs(x)<=world.r_o:
        if ((check_if_on_circle(world,x,np.abs(y)-world.track_len/2))and np.abs(y)>=world.track_len/2) or check_if_on_straight(world,x,y):
            return True
    return False
