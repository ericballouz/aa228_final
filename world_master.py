import numpy as np

class World:
    def __init__(self):
        self.r_o = 250 #outer radius
        self.r_i = (3/4)*self.r_o #inner_radius
        self.window_w = 4*self.r_o
        self.window_l = 8*self.r_o
        self.track_len = 3*self.r_o
        self.num_checkpts = 24
        checkpt_y_size = (3*self.r_o)/((self.num_checkpts/2)-4)
        checkpt_y_list = list()
        checkpt_y_list.append(-1.5*self.r_o-self.r_i)
        for i in range(0,int(self.num_checkpts/2)-4):
            y_bottom_of_straight = -1.5*self.r_o
            checkpt_y_list.append(y_bottom_of_straight + i*checkpt_y_size)
        checkpt_y_list.append(1.5*self.r_o)
        checkpt_y_list.append(1.5 * self.r_o + self.r_i)
        self.checkpt_y_array = np.asarray(checkpt_y_list)
        self.checkpt_x_array = np.asarray([0])
        self.start_x,self.start_y,self.start_V,self.start_theta = self.get_start_state()
        self.start_state = tuple([self.start_x,self.start_y,self.start_V,self.start_theta])
        self.checkpts_hit = [0]
        #self.state = (self.x,self.y,self.V,self.theta)

    def get_start_state(self):
        x_start = self.r_i + ((self.r_o - self.r_i) / 2)
        y_start = -1
        V_start = 0  # stopped
        theta_start = 0  # facing up
        return tuple([x_start, y_start, V_start, theta_start])

    def check_if_car_in_world(self, state):
        (x,y,V,theta) = state
        if np.abs(x) <= self.window_w / 2 and np.abs(y) <= self.window_l / 2:
            return True
        return False

    def check_if_on_circle(self,state):
        (x, y, V, theta) = state
        loc = (x * x) + (np.abs(y)-self.track_len/2) * (np.abs(y)-self.track_len/2)
        if loc >= (self.r_i * self.r_i) and loc <= (self.r_o * self.r_o):
            return True
        return False

    def check_if_on_straight(self,state):
        (x, y, V, theta) = state
        if np.abs(x) >= self.r_i and np.abs(y)<= self.track_len / 2:
            return True
        return False

    def check_if_car_on_track(self,state):
        (x, y, V, theta) = state
        if np.abs(x) <= self.r_o:
            if (self.check_if_on_circle(state)) and (np.abs(y)>= self.track_len/2) or self.check_if_on_straight(state):
                return True
        return False

    def get_curr_checkpt(self,state):
        ''''
        decided to hardcode in checkpt numbers
        '''
        y_checkpts = self.checkpt_y_array
        x_checkpts = self.checkpt_x_array
        (x, y, V, theta) = state
        if not self.check_if_car_on_track(state):
            return None
        if x > x_checkpts[0]:
            right = True
        else:
            right = False
        if y < y_checkpts[0]:
            if right:
                return 19
            return 18
        if y < y_checkpts[1]:
            if right:
                return 20
            return 17
        if y < y_checkpts[2]:
            if right:
                return 21
            return 16
        if y < y_checkpts[3]:
            if right:
                return 22
            return 15
        if y < y_checkpts[4]:
            if right:
                return 23
            return 14
        if y < y_checkpts[5]:
            if right:
                return 0
            return 13
        if y < y_checkpts[6]:
            if right:
                return 1
            return 12
        if y < y_checkpts[7]:
            if right:
                return 2
            return 11
        if y < y_checkpts[8]:
            if right:
                return 3
            return 10
        if y < y_checkpts[9]:
            if right:
                return 4
            return 9
        if y < y_checkpts[10]:
            if right:
                return 5
            return 8
        if y>y_checkpts[y_checkpts.size-1]:
            if right:
                return 6
            return 7

    def update_checkpts_seen(self,state):
        curr_checkpt = self.get_curr_checkpt(state)
        #print(self.checkpts_hit)
        if (curr_checkpt is not None) and curr_checkpt not in self.checkpts_hit: #UPDATEd THIS SO THAT IT GOES IN ORDER (1 more than prev)

            if curr_checkpt==1+self.checkpts_hit[len(self.checkpts_hit)-1]: #if empty or if next one is 1 more than current max
                #print("Just saw checkpt {}".format(curr_checkpt))
                self.checkpts_hit.append(curr_checkpt)
                return True
        return False

    def seen_all_checkpts(self):
        checkpt_list = self.checkpts_hit
        if sorted(checkpt_list) == checkpt_list and len(checkpt_list) == self.num_checkpts:
            return True
        return False

    def successfully_finished(self,state):
        if self.seen_all_checkpts() and self.check_if_car_on_track(state) and self.get_curr_checkpt(state) == 0:
            return True
        return False

    def printFarthestCheckpt(self):
        print("Farthest checkpoint: %d" % max(self.checkpts_hit))
        return
