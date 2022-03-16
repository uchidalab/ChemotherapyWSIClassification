import os
import cv2
import glob
from PIL import Image
from natsort import natsorted
from util import num_to_color, get_filename, convert_space2uline


"""
cg_dir: cancergrade_mask directory
"""


def make_overlaid(
    cg_dir: str,
    bg_dir: str,
    output_dir: str,
    th: int=240,
    labels: list=['Tumor bed', 'Residual tumor'],
    is_nolabel_normal: bool=False
):
    if is_nolabel_normal:
        nolabel_color = num_to_color(0)
        nolabel_grade = 0
    else:
        nolabel_color = (0, 0, 0)
        nolabel_grade = len(labels) + 1

    bg_paths = natsorted(glob.glob(bg_dir + "*_mask_level05_bg.tif"))
    for bg_path in bg_paths:
        fn = get_filename(bg_path)
        wsi_name = fn.replace("_mask_level05_bg", "")
        print(wsi_name)

        # make canvas from bg_mask
        canvas = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        canvas_gray = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)

        # if cancer grade mask exists
        if os.path.isfile(cg_dir + wsi_name + "_mask_level05_nolabel.tif"):
            # paste each cancergrade's color and num
            for i, label in enumerate(labels):
                grade = i + 1
                label = convert_space2uline(label)
                grade_path = cg_dir + wsi_name + "_mask_level05_" + label + ".tif"
                grade_mask = cv2.imread(grade_path, cv2.IMREAD_GRAYSCALE)
                canvas[grade_mask > th] = num_to_color(grade)
                canvas_gray[grade_mask > th] = grade

            # paste nolabel area
            nolabel_path = cg_dir + wsi_name + "_mask_level05_nolabel.tif"
            nolabel_img = cv2.imread(nolabel_path, cv2.IMREAD_GRAYSCALE)
            canvas[nolabel_img > th] = nolabel_color
            canvas_gray[nolabel_img > th] = nolabel_grade

            Image.fromarray(canvas).save(output_dir + "rgb/" + wsi_name + "_overlaid.tif")
            Image.fromarray(canvas_gray).save(
                output_dir + "gray/" + wsi_name + "_overlaid.tif"
            )
        else:
            print(f"{wsi_name}: cancer grade mask does not exists")


if __name__ == "__main__":
    PARENT_DIR = "/mnt/ssdwdc/ResearchData/chemotherapy/202109_chemotherapy/"

    CANCERGRADE_DIR = PARENT_DIR + "mask_cancergrade/"
    BG_MASK_DIR = PARENT_DIR + "mask_bg/"
    OVERLAID_DIR = PARENT_DIR + "overlaid_[0, 1, 2]/"

    make_overlaid(CANCERGRADE_DIR, BG_MASK_DIR, OVERLAID_DIR)
