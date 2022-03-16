import os
import sys

chemotherapy_label_dict = {
    "Tumor bed": "Tumor bed",
    "Tumor range": "Tumor bed",
    "Tumor Range": "Tumor bed",
    "Tumor Bed": "Tumor bed",
    "Residual tumor": "Residual tumor",
    "Residual Tumor": "Residual tumor",
}


def f_write(file_path, s):
    with open(file_path, mode="a") as f:
        f.write(s + "\n")


def color_to_num(color):
    if color == (200, 200, 200):
        num = 0
    elif color == (0, 65, 255):
        num = 1
    elif color == (255, 40, 0):
        num = 2
    elif color == (53, 161, 107):
        num = 3
    elif color == (250, 245, 0):
        num = 4
    elif color == (102, 204, 255):
        num = 5
    else:
        sys.exit("invalid color:" + str(color))
    return num


def num_to_color(num):
    if isinstance(num, list):
        num = num[0]

    if num == 0:
        color = (200, 200, 200)
    elif num == 1:
        color = (0, 65, 255)
    elif num == 2:
        color = (255, 40, 0)
    elif num == 3:
        color = (53, 161, 107)
    elif num == 4:
        color = (250, 245, 0)
    elif num == 5:
        color = (102, 204, 255)
    else:
        sys.exit("invalid number:" + str(num))
    return color


def get_filename(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[0]


def convert_space2uline(text: str):
    if " " in text:
        text = text.replace(" ", "_")
    return text
