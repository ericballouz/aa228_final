from create_track import World

world = World()

def get_start_state():
    x_start = world.r_i + ((world.r_o-world.r_i)/2)
    y_start = -100
    V_start = 0 #stopped
    theta_start = 0 #facing up
    return tuple([x_start, y_start, V_start, theta_start])
