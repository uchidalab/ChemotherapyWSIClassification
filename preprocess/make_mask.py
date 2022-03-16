import os
import sys
import glob
import shutil
import numpy as np
import cv2
from PIL import Image
from natsort import natsorted


def cleanup_directory(name):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name)


def update_canvases(image_mask, canvas, canvas_gray, color, i, th=250):
    mask = cv2.imread(image_mask, cv2.IMREAD_GRAYSCALE)
    canvas[mask > th] = 6
    canvas_gray[mask > th] = i
    return canvas, canvas_gray


def extract_mask_path(src_list, keyword="grade00"):
    l = [path for path in src_list if keyword in path]
    return l[0]


def overlaid_mask_image(PATH_WORK, CLASSES, th=250):
    cleanup_directory(PATH_WORK + "mask_cancergrade/overlaid_" + str(CLASSES) + "/")
    cleanup_directory(
        PATH_WORK + "mask_cancergrade_gray/overlaid_" + str(CLASSES) + "/"
    )

    images_full = natsorted(os.listdir(PATH_WORK + "origin/"))
    images = []
    for image_full in images_full:
        image, _ = os.path.splitext(image_full)
        images.append(image)

    for image in images:
        image_masks = glob.glob(
            PATH_WORK + "mask_cancergrade/*" + image + "_mask" + "*.tif"
        )
        image_masks.sort()

        for image_mask in image_masks:
            if image_mask.find("bg") >= 0:  # bg
                bg_name = image_mask

        bg_mask = cv2.imread(bg_name, cv2.IMREAD_GRAYSCALE)  # サイズ確認用 0じゃなくても，なんでもいい
        canvas = np.zeros((bg_mask.shape[0], bg_mask.shape[1], 3))
        canvas[bg_mask > th] = (255, 255, 255)

        canvas_gray = np.zeros((bg_mask.shape[0], bg_mask.shape[1])) + 255

        for (i, cls) in enumerate(CLASSES):
            if isinstance(cls, list):
                for SUB_CLASS in cls:
                    image_mask = extract_mask_path(
                        image_masks, keyword=f"grade{SUB_CLASS:02d}")
                    color = num_to_color(cls[0])
                    canvas, canvas_gray = \
                        update_canvases(image_mask, canvas, canvas_gray, color, i, th=th)
            else:
                image_mask = extract_mask_path(
                    image_masks, keyword=f"grade{cls:02d}")
                color = num_to_color(cls)
                canvas, canvas_gray = \
                    update_canvases(image_mask, canvas, canvas_gray, color, i, th=th)

        Image.fromarray(canvas.astype(np.uint8)).save(
            PATH_WORK
            + "mask_cancergrade/overlaid_"
            + str(CLASSES)
            + "/"
            + image
            + "_overlaid.tif"
        )
        Image.fromarray(canvas_gray.astype(np.uint8)).save(
            PATH_WORK
            + "mask_cancergrade_gray/overlaid_"
            + str(CLASSES)
            + "/"
            + image
            + "_overlaid.tif"
        )


if __name__ == "__main__":
    PATH_WORK = "/mnt/secssd/SSDA_Annot_WSI_strage/mnt1/MF0003/"
    CLASSES = [0, 1, 2, 3, 4]

    overlaid_mask_image(PATH_WORK, CLASSES, th=250)