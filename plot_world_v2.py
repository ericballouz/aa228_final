import matplotlib.pyplot as plt
#from create_track import World
import numpy as np
#import start_state


def plot_boundary(world):
    # create boundary
    x_space = np.linspace(-world.window_w / 2, world.window_w / 2)
    y_space = np.linspace(-world.window_l / 2, world.window_l / 2)
    left_side = -world.window_w / 2 * np.ones(y_space.size)
    right_side = world.window_w / 2 * np.ones(y_space.size)
    bottom_side = -world.window_l / 2 * np.ones(x_space.size)
    top_side = world.window_l / 2 * np.ones(x_space.size)
    plt.figure()
    plt.plot(left_side, y_space, 'k')
    plt.plot(right_side, y_space, 'k')
    plt.plot(x_space, bottom_side, 'k')
    plt.plot(x_space, top_side, 'k')



def plot_track(world):
    # create track
    y_track_space = np.linspace(-world.track_len / 2, world.track_len / 2, num=1000)
    outside_left = -world.r_o * np.ones(y_track_space.size)
    outside_right = world.r_o * np.ones(y_track_space.size)
    inside_left = -world.r_i * np.ones(y_track_space.size)
    inside_right = world.r_i * np.ones(y_track_space.size)

    x_outer_circle_top = np.linspace(-world.r_o, world.r_o, num=1000)
    x_inner_circle_top = np.linspace(-world.r_i, world.r_i, num=1000)
    x_circle_top_size = x_outer_circle_top.size
    y_outer_circle_top = list()
    y_inner_circle_top = list()

    y_outer_circle_bottom = list()
    y_inner_circle_bottom = list()
    for i in range(x_circle_top_size):
        y_outer_circle_top.append(
            (world.track_len / 2) + np.sqrt(world.r_o * world.r_o - x_outer_circle_top[i] * x_outer_circle_top[i]))
        y_inner_circle_top.append(
            (world.track_len / 2) + np.sqrt(world.r_i * world.r_i - x_inner_circle_top[i] * x_inner_circle_top[i]))
        y_outer_circle_bottom.append(
            -(world.track_len / 2) - np.sqrt(world.r_o * world.r_o - x_outer_circle_top[i] * x_outer_circle_top[i]))
        y_inner_circle_bottom.append(
            -(world.track_len / 2) - np.sqrt(world.r_i * world.r_i - x_inner_circle_top[i] * x_inner_circle_top[i]))

    y_outer_circle_top = np.asarray(y_outer_circle_top)
    y_inner_circle_top = np.asarray(y_inner_circle_top)
    y_outer_circle_bottom = np.asarray(y_outer_circle_bottom)
    y_inner_circle_bottom = np.asarray(y_inner_circle_bottom)

    plt.plot(outside_left, y_track_space, 'b')
    plt.plot(outside_right, y_track_space, 'b')
    plt.plot(inside_left, y_track_space, 'b')
    plt.plot(inside_right, y_track_space, 'b')

    plt.plot(x_outer_circle_top, y_outer_circle_top, 'b')
    plt.plot(x_inner_circle_top, y_inner_circle_top, 'b')
    plt.plot(x_outer_circle_top, y_outer_circle_bottom, 'b')
    plt.plot(x_inner_circle_top, y_inner_circle_bottom, 'b')



def plot_checkpts(world):
    # plot checkpoints
    checkpt_y_array = world.checkpt_y_array
    for i in range(checkpt_y_array.size):
        y_val = checkpt_y_array[i]
        x_checkpt_left = np.linspace(-world.r_o, -world.r_i, num=1000)
        y_checkpt_left = y_val * np.ones(x_checkpt_left.size)
        plt.plot(x_checkpt_left, y_checkpt_left, 'r')
        x_checkpt_right = np.linspace(world.r_i, world.r_o, num=1000)
        y_checkpt_right = y_val * np.ones(x_checkpt_right.size)
        plt.plot(x_checkpt_right, y_checkpt_right, 'r')
    # plot vertical checkpts
    vertical_checkpt_y_top = np.linspace(1.5 * world.r_o + world.r_i, 2.5 * world.r_o, num=1000)
    vertical_checkpt_x_top = np.zeros(vertical_checkpt_y_top.size)
    plt.plot(vertical_checkpt_x_top, vertical_checkpt_y_top, 'r')
    vertical_checkpt_y_bottom = np.linspace(-2.5 * world.r_o, -1.5 * world.r_o - world.r_i, num=1000)
    plt.plot(vertical_checkpt_x_top, vertical_checkpt_y_bottom, 'r')


def plot_starting_pt(world):
    # plot starting point
    x_start, y_start, V_start, theta_start = world.get_start_state()
    plt.scatter(x_start, y_start)

def plot_startup(world):
    plot_boundary(world)
    plot_track(world)
    plot_checkpts(world)
    plot_starting_pt(world)

def plot_trajectory(x_array,y_array):
    plt.scatter(x_array,y_array)


def show_plot():
    plt.show()

