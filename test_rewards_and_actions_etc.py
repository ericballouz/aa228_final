import numpy as np
from world_master import World
import model
import plot_world_v2


dt = 0.1




def main():
    world = World() #initialize world
    plot_world_v2.plot_startup(world)
    R_start = model.R(world, world.start_state)
    A_space = model.action_space()
    #print(A_space)
    s = world.start_state
    #print(s)
    x_visited = list()
    y_visited = list()
    r_total = 0
    for i in range(100000):
        a = A_space[np.random.randint(0,len(A_space))]
        next_state = model.nextState(s, a, dt)
        #print(next_state)
        if world.check_if_car_on_track(next_state):
            s = next_state
            x, y, v, theta = s
            r = model.R(world,s)
            #print("s is {}, r is {}, and curr_checkpt is {}".format(x,r,world.get_curr_checkpt(s)))
            r_total+=r[0]
            x_visited.append(x)
            y_visited.append(y)
    x_visited = np.asarray(x_visited)
    y_visited = np.asarray(y_visited)
    #print(y_visited)

    plot_world_v2.plot_trajectory(x_visited,y_visited)
    plot_world_v2.show_plot()

if __name__ == '__main__':
    main()
