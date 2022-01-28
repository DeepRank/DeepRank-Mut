#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import glfw
from OpenGL.GL import *
import numpy
import h5py
import pyrr


# Assure that python can find the deeprank files:
deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deeprank_root)

from deeprank.operate.hdf5data import load_grid_center, load_grid_points, load_grid_data


arg_parser = ArgumentParser(description="visualize a 3D grid")
arg_parser.add_argument("hdf5_path", help="path to hdf5 file, containing the grid")
arg_parser.add_argument("entry_name", help="name of the entry, containing the grid")
arg_parser.add_argument("feature_name", help="name of the grid feature, to visualize")


class Grid:
    "represents the grid with feature values and a center"

    def __init__(self, center, xs, ys, zs, feature_data):
        self.center = center
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.feature_data = feature_data

        self.max_value = numpy.max(feature_data)
        self.min_value = numpy.min(feature_data)


def get_grid(hdf5_file, entry_name, feature_name):
    "gets the grid from the hdf5 file"

    if entry_name not in hdf5_file:
        raise ValueError(f"No such entry: {entry_name}")

    variant_group = hdf5_file[entry_name]

    xs, ys, zs = load_grid_points(variant_group)
    center = load_grid_center(variant_group)
    feature_data = load_grid_data(variant_group, "Feature_ind")

    if feature_name not in feature_data:
        raise ValueError(f"No such feature: {feature_name}")

    return Grid(center, xs, ys, zs, feature_data[feature_name])


def render_grid_point(index_x, index_y, index_z, grid):
    "procedure to draw one grid point"

    value = grid.feature_data[index_x, index_y, index_z]
    if value == 0.0:
        glColor4f(0.0, 0.0, 0.0, 0.0)

    if value < 0.0:
        value = value / grid.min_value
        glColor4f(1.0, 0.0, 0.0, value)
    else:
        value = value / grid.max_value
        glColor4f(0.0, 0.0, 1.0, value)

    glVertex3f(grid.xs[index_x], grid.ys[index_y], grid.zs[index_z]);


def render_grid(grid):
    "draws the entire grid on the screen"

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBegin(GL_LINES)

    #  lines along the x-axis
    for index_z in range(len(grid.zs)):
        for index_y in range(len(grid.ys)):
            for index_x in range(len(grid.xs)):
                if index_x > 0:
                    render_grid_point(index_x - 1, index_y, index_z, grid)
                    render_grid_point(index_x, index_y, index_z, grid)

    # lines along the y-axis
    for index_x in range(len(grid.xs)):
        for index_z in range(len(grid.zs)):
            for index_y in range(len(grid.ys)):
                if index_y > 0:
                    render_grid_point(index_x, index_y - 1, index_z, grid)
                    render_grid_point(index_x, index_y, index_z, grid)

    # lines along the z-axis
    for index_x in range(len(grid.xs)):
        for index_y in range(len(grid.ys)):
            for index_z in range(len(grid.zs)):
                if index_z > 0:
                    render_grid_point(index_x, index_y, index_z - 1, grid)
                    render_grid_point(index_x, index_y, index_z, grid)

    glEnd()

if __name__ == "__main__":

    args = arg_parser.parse_args()

    # load the grid
    with h5py.File(args.hdf5_path, 'r') as f5:
        grid = get_grid(f5, args.entry_name, args.feature_name)

    screen_width = 800
    screen_height = 600

    # initializing glfw
    glfw.init()

    # creating a window with 800 width and 600 height
    window = glfw.create_window(screen_width, screen_height,"Grid View", None, None)
    glfw.set_window_pos(window, 400, 200)
    glfw.make_context_current(window)

    #create a perspective matrix
    aspect_ratio = screen_width / screen_height
    projection_matrix = pyrr.matrix44.create_perspective_projection(45.0, aspect_ratio, 0.1, 1000.0)

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projection_matrix);

    # setting color for background
    glClearColor(1.0, 1.0, 1.0, 1.0)

    eye_direction = numpy.array([0.0, 0.0, 1.0])  # rotatable vector pointing from the grid center to the camera
    up_direction = numpy.array([0.0, 1.0, 0.0])  # up direction for the camera

    zoom = 50.0

    previous_mouse_x, previous_mouse_y = glfw.get_cursor_pos(window)
    while not glfw.window_should_close(window):

        glfw.poll_events()

        # Determine whether the mouse button is down and how much it moved.
        mouse_button_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        mouse_motion_x = mouse_x - previous_mouse_x
        mouse_motion_y = mouse_y - previous_mouse_y
        previous_mouse_x = mouse_x
        previous_mouse_y = mouse_y

        if mouse_button_down:
            # rotate the eye direction vector

            mouse_motion_axis = numpy.array([mouse_motion_y, -mouse_motion_x, 0.0])
            amount_scrolled = numpy.sqrt(numpy.square(mouse_motion_x) + numpy.square(mouse_motion_y))

            if amount_scrolled > 0.0:
                rotation_matrix = pyrr.matrix33.create_from_axis_rotation(mouse_motion_axis, 0.05 * amount_scrolled)
                eye_direction = pyrr.matrix33.apply_to_vector(rotation_matrix, eye_direction)
                eye_direction = eye_direction / numpy.linalg.norm(eye_direction)

        view_matrix = pyrr.matrix44.create_look_at(grid.center - eye_direction * zoom,
                                                   grid.center, up_direction)

        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(view_matrix);

        render_grid(grid)

        glfw.swap_buffers(window)

    glfw.terminate()
