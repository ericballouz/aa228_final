from create_track import World
import check_if_on_track

world = World()
y_checkpts = world.checkpt_y_array
x_checkpts = world.checkpt_x_array
num_checkpts = y_checkpts * x_checkpts



def get_curr_checkpt(state):
    ''''
    decided to hardcode in checkpt numbers
    '''
    (x, y, V, theta) = state
    if not check_if_on_track.check_if_car_on_track(state):
        return None
    if x>x_checkpts[0]:
        right = True
    else:
        right = False
    if y<y_checkpts[0]:
        if right:
            return 10
        return 9
    if y>y_checkpts[y_checkpts.size-1]:
        if right:
            return 3
        return 4
    if y<y_checkpts[1]:
        if right:
            return 11
        return 8
    if y<y_checkpts[2]:
        if right:
            return 0
        return 7
    if y<y_checkpts[3]:
        if right:
            return 1
        return 6
    if y<y_checkpts[4]:
        if right:
            return 2
        return 5


def update_checkpts_seen(state,checkpt_list):
    curr_checkpt = get_curr_checkpt(state)
    if curr_checkpt is not None and curr_checkpt not in checkpt_list:
        checkpt_list.append(curr_checkpt)


def seen_all_checkpts(checkpt_list):
    if sorted(checkpt_list) == checkpt_list and len(checkpt_list)==world.num_checkpts:
        return True
    return False

