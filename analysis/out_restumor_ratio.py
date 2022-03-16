import os
import sys
import cv2
import numpy as np
import yaml
from natsort import natsorted
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from annotation.util import color_to_num, num_to_color
from src.util import fix_seed


def export_log(wsi_name: str, ratio: float, file_path: str):
    with open(file_path, mode='a') as f:
        f.write(f"[{wsi_name}]: {ratio:.04f}\n")


def calculate_ratio(imgmap_path: str, log_file: str):
    wsi_name = os.path.splitext(os.path.basename(imgmap_path))[0]
    print(f"== {wsi_name} ==")
    img = cv2.imread(imgmap_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray_canvas = np.full((img.shape[0], img.shape[1]), 256, dtype=np.uint8)

    # color to num
    gray_canvas[(img == num_to_color(1)).all(axis=-1)] = 1
    gray_canvas[(img == num_to_color(2)).all(axis=-1)] = 2

    # count tumor region pixels (class 1)
    tr = np.sum(gray_canvas == 1)

    # count residual tumor pixels (class 2)
    rt = np.sum(gray_canvas == 2)

    ratio = rt / (tr + rt)
    print(f"ResidualTumor ratio: {ratio:.04f}")
    export_log(wsi_name, ratio, file_path=log_file)
    return ratio


def main():
    truemap_dir = "/mnt/ssdsam/chemotherapy_strage/mnt1/mask_cancergrade/overlaid_[0, 1, 2]/"
    true_log_path = "/mnt/ssdsam/chemotherapy_strage/result_LEV012/predmap/residual_tumor_ratio_truemap.txt"
    truemap_list = [truemap_dir + fname for fname in natsorted(os.listdir(truemap_dir))]

    predmap_dir = "/mnt/ssdsam/chemotherapy_strage/result_LEV012/predmap/cv2/"
    pred_log_path = "/mnt/ssdsam/chemotherapy_strage/result_LEV012/predmap/cv2/residual_tumor_ratio_predmap_LEV012_cv2.txt"
    predmap_list = [predmap_dir + fname for fname in natsorted(os.listdir(predmap_dir)) if "_cl" not in fname]

    print("calculate predmap's ratio")
    for path in predmap_list:
       calculate_ratio(path, pred_log_path)

    # print("calculate truemap's ratio")
    # for path in truemap_list:
    #    calculate_ratio(path, true_log_path)


if __name__ == "__main__":
    main()