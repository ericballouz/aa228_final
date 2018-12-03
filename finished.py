from create_track import World
import check_if_on_track
import checkpts

world = World()

def is_finished(state,checkpts_list):
    if checkpts.seen_all_checkpts(checkpts_list) and check_if_on_track.check_if_car_on_track(state) and checkpts.get_curr_checkpt(state)==0:
        return True
    return False