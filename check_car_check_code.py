import numpy as np
import check_if_on_track
import matplotlib.pyplot as plt
from create_track import World

world = World()

def sample_pts():
    x_list = list()
    y_list = list()
    for i in range(100000):
        x = np.random.uniform(-world.window_w/2,world.window_w/2)
        y = np.random.uniform(-world.window_l/2,world.window_l/2)
        print(x,y)
        if check_if_on_track.check_if_car_on_track((x,y,0,0)):
            x_list.append(x)
            y_list.append(y)
    return np.asarray(x_list),np.asarray(y_list)

plt.figure(figsize=(5,5))
x_array, y_array = sample_pts()
plt.scatter(x_array,y_array)
x_min = -world.window_w/2
x_max = world.window_w/2
y_min = -world.window_l/2
y_max = world.window_l/2
plt.axis([x_min, x_max, y_min, y_max])
plt.show()

