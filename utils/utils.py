
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

path = 'utils/img/'
# xmin, xmax = -74.03, -73.89
# ymin, ymax = 40.69, 40.89

xmin, xmax = 100, 5
ymin, ymax = 100, 5

def vrp_action_go_from_a_to_b(a, b):
    # 0: Up, 1: Down, 2: Left, 3: Right
    action = 0
    cur_x = a[0]
    cur_y = a[1]
    tar_x = b[0]
    tar_y = b[1]

    x_diff = tar_x - cur_x
    y_diff = tar_y - cur_y

    if abs(x_diff) >= abs(y_diff):
        # Move horizontally
        if x_diff > 0:
            action = 4
        elif x_diff < 0:
            action = 3
    else:
        # Move vertically
        if y_diff > 0:
            action = 1
        elif y_diff < 0:
            action = 2

    return action



def draw_map(file_name):
    with open(file_name, "rb") as f:
        load = pickle.load(f)
    fig, ax = plt.subplots(figsize=(10, 10))
    load[1].plot(ax=ax,alpha=0.2)
    fig.show()
    fig.canvas.draw()
    return fig, ax


def draw_image(file_name):
    im = Image.open(path+file_name).convert("RGBA")
    width, height = im.size
    scale_x = width / (xmax - xmin)
    scale_y = height / (ymax - ymin)
    scale = [scale_x, scale_y]
    return im, scale


def draw_vehicle(image, pos, scale, fill = '#ff6361', r=6):
    scale_x = scale[0]
    scale_y = scale[1]
    y = image.size[1] - (pos[0]-ymin)*scale_y
    x = (pos[1]-xmin)*scale_x
    x_dash = x + r
    y_dash = y + r
    ImageDraw.Draw(image).ellipse([(x-r, y-r), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_order(image, pos, scale, fill='#003f5c', r = 4):
    scale_x = scale[0]
    scale_y = scale[1]
    y = image.size[1] - (pos[0]-ymin)*scale_y
    x = (pos[1]-xmin)*scale_x
    x_dash = x + r
    y_dash = y + r
    ImageDraw.Draw(image).rectangle([(x-r, y-r), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_map_h3(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    df[['h3_res9', 'geometry']].plot(alpha=0.3, ax=ax, facecolor="none",edgecolor='#444e86')

    # fig.show()
    # fig.canvas.draw()
    return fig, ax


def load_file(file_name):
    with open(path+file_name, 'rb') as pickle_file:
        h3 = pickle.load(pickle_file)
    return h3

def fill_cell_im(image, icon, pos, cell_size=None, fill='black', margin=0):
    # assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    # margin *= cell_size
    # x, y, x_dash, y_dash = row + margin, col + margin, row + cell_size - margin, col + cell_size - margin
    # ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)
    # image.paste(icon, (int(row)+1, int(col)+1), icon)
    image.paste(icon, (int(row), int(col)), icon)