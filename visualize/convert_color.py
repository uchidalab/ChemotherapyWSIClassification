import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import glob
import os


def change_color(img, source_colors, target_colors):
    canvas = img.copy()
    
    for i in range(len(source_colors)):
        src = source_colors[i] 
        trg = target_colors[i]
        print(img.shape)
        mask = np.all(img == src, axis=-1)
        canvas[mask] = trg
    return canvas

def src_num_to_color(num):
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


if __name__ == "__main__":
    source_colors = [(0, 65, 255), (255, 40, 0)]
    target_colors = [ (255, 40, 0), (0, 65, 255)]

    img_path_list = glob.glob("/mnt/ssdsam/chemotherapy_strage/result_LEV012/predmap/*.png")
    output_dir = "/mnt/ssdsam/chemotherapy_strage/result_LEV012/predmap/"

    for input_path in img_path_list:
        img = cv2.imread(input_path)  # BGR
        bf_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # RGBに変換
        af_img = change_color(bf_img, source_colors, target_colors)  # RGB

        output_path = output_dir + os.path.basename(input_path)
        print(output_path)

        cv2.imwrite(output_path, cv2.cvtColor(af_img, cv2.COLOR_BGR2RGB))