import numpy as np

class World:
    def __init__(self):
        self.r_o = 250 #outer radius
        self.r_i = (3/4)*self.r_o #inner_radius
        self.window_w = 4*self.r_o
        self.window_l = 8*self.r_o
        self.track_len = 3*self.r_o
        self.num_checkpts = 12
        checkpt_y_size = (3*self.r_o)/((self.num_checkpts/2)-2)
        checkpt_y_list = list()
        for i in range(0,int(self.num_checkpts/2)-2):
            y_bottom_of_straight = -1.5*self.r_o
            checkpt_y_list.append(y_bottom_of_straight + i*checkpt_y_size)
        checkpt_y_list.append(1.5*self.r_o)
        self.checkpt_y_array = np.asarray(checkpt_y_list)
        self.checkpt_x_array = np.asarray([0])

