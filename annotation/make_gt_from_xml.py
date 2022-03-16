# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import glob
from openslide import OpenSlide
from PIL import Image, ImageDraw

# from lxml import etree
from xml.etree import ElementTree as ET
from natsort import natsorted

from util import f_write, get_filename, convert_space2uline


def init_annot_dict(labels:list):
    result = {}
    for label in labels:
        result[str(label)] = []
    return result


# --------------------#
#   load xml-file     #
#     (for ASAP)      #
# --------------------#
def load_xml_file(
    xml_path, labels:list=['Tumor bed', 'Residual tumor'], label_dict:dict=None, log_path=None
):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = root[0].findall("Annotation")
    num_annotations = len(annotations)

    annot_dict = init_annot_dict(labels)

    for i in range(num_annotations):
        tmp = annotations[i].findall("Coordinates")
        dots = tmp[0].findall("Coordinate")
        num_dots = len(dots)
        dots_array = np.zeros((num_dots, 2), dtype=np.uint32)

        if num_dots > 1:
            for j in range(num_dots):
                x = int(round(float(dots[j].attrib["X"])))
                y = int(round(float(dots[j].attrib["Y"])))

                dots_array[j, 0] = x
                dots_array[j, 1] = y

            name = annotations[i].attrib["Name"]
            if label_dict is not None:
                try:
                    name = label_dict[name]
                except KeyError:
                    print(f"Cannot find {name} in label_dict")

            if name in labels:
                annot_dict[name].append(dots_array)
            else:
                try:
                    print(name)
                    if log_path:
                        f_write(log_path, "{}; type: {}".format(xml_path, name))
                except:
                    print("failed to get annotation name")
                    print("annotation_number: {}".format(i))
        else:
            print("only include 1 coordinates")

    return annot_dict


# --------------------#
#   load ndpa-file    #
#     (for NDPA)      #
# --------------------#
def calc_ndpa_coord(wsi_path, x, y):
    wsi_name = os.basename(wsi_path)
    print(wsi_name)

    wsi = OpenSlide(wsi_path)

    mppx = float(wsi.properties["openslide.mpp-x"])
    mppy = float(wsi.properties["openslide.mpp-y"])

    # use it for μm -> nm
    dx, dy = mppx * 10 ** 3, mppy * 10 ** 3

    xofset_slide_center = float(wsi.properties["hamamatsu.XOffsetFromSlideCentre"])
    yofset_slide_center = float(wsi.properties["hamamatsu.YOffsetFromSlideCentre"])
    Xoffset = xofset_slide_center / dx
    Yoffset = yofset_slide_center / dy

    width = wsi.level_dimensions[0][0]
    height = wsi.level_dimensions[0][1]
    # print("(w, h)=({}, {})".format(width, height))

    cw, ch = width / 2, height / 2
    epx, epy = x / dx + cw - Xoffset, y / dy + ch - Yoffset
    edit_points = [epx, epy]
    # print("edit_points: {}".format(edit_points))

    return edit_points


def load_ndpa_file(
    ndpa_path, wsi_path, labels:list=['Tumor bed', 'Residual tumor']
):
    tree = ET.parse(ndpa_path)
    root = tree.getroot()

    annotations = root.findall("ndpviewstate")
    num_annotations = len(annotations)
    print(num_annotations)

    annot_dict = init_annot_dict(labels)

    for i in range(num_annotations):
        tmp = annotations[i]
        dots = tmp.find("annotation/pointlist").findall("point")
        num_dots = len(dots)
        print(num_dots)
        dots_array = np.zeros((num_dots, 2), dtype=np.uint32)

        for j in range(num_dots):
            x = int(round(float(dots[j].find("x").text)))
            y = int(round(float(dots[j].find("y").text)))

            points = calc_ndpa_coord(wsi_path, x, y)

            dots_array[j, 0] = points[0]
            dots_array[j, 1] = points[1]

        name = tmp.find("title").text
        print(name)

        if name in labels:
            annot_dict[name].append(dots_array)
        else:
            try:
                print(name)
            except:
                print("failed to get annotation name")
                print("annotation_number: {}".format(i))

    return annot_dict


# ------------------------------#
#    draw cancergrade mask     #
# ------------------------------#
def draw_cancergrade_mask(
    title: str, annot_list: list, org_size: tuple, output_size: tuple, save_dir: str
):
    """
    annot_list:
        [dots_array_00, ... , dots_array_0n]
    """
    mask = Image.new("L", org_size)
    mask_draw = ImageDraw.Draw(mask)

    # output black-mask, if annot_list is empty
    if len(annot_list) > 0:
        # one of the cluster of the class
        for i in range(len(annot_list)):
            annot_tuple = annot_list[i].flatten().tolist()
            mask_draw.polygon(annot_tuple, fill=255, outline=None)

    output_mask = mask.resize(output_size, Image.LANCZOS)
    output_mask.save(save_dir + title + ".tif")


# ---------------------------------#
#  make pre-cancergrade from xml  #
# ---------------------------------#
def make_pre_cancergrade(
    wsi_path: str, annot_dict: dict, save_dir: str, level: int=5, wsi_name: str=None,
):
    """
    annot_dict:
        { "Tumor": [dots_array_00, ... , dots_array_0n],
        "Non-Tumor": [dots_array_10, ... , dots_array_1n] }
        level: rescaled level
    """
    if not wsi_name:
        # wsi_name = return_filename(wsi_path)
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        print(wsi_name)

    # open whole-slide image
    wsi = OpenSlide(wsi_path)

    # get whole-slide properties
    width = wsi.level_dimensions[0][0]
    height = wsi.level_dimensions[0][1]
    org_size = (width, height)
    print("wsi-size: ({}, {})".format(width, height))

    # get resized-image size
    resized_width = wsi.level_dimensions[level][0]
    resized_height = wsi.level_dimensions[level][1]
    output_size = (resized_width, resized_height)
    print("output-size: ({}, {})".format(resized_width, resized_height))

    for label, annot_list in annot_dict.items():
        label = convert_space2uline(label)
        title = f"{wsi_name}_mask_level0{level}_{label}"
        draw_cancergrade_mask(title, annot_list, org_size, output_size, save_dir)

    #  # =================================== #
    # normal, hsil, lsil, scc, others, etc = annot_lists

    # title = wsi_name + "_mask_level0" + str(level) + "_grade00"
    # draw_cancergrade_mask(title, normal, org_size, output_size, SAVE_DIR)

    # title = wsi_name + "_mask_level0" + str(level) + "_grade01"
    # draw_cancergrade_mask(title, hsil, org_size, output_size, SAVE_DIR)

    # title = wsi_name + "_mask_level0" + str(level) + "_grade02"
    # draw_cancergrade_mask(title, lsil, org_size, output_size, SAVE_DIR)

    # title = wsi_name + "_mask_level0" + str(level) + "_grade03"
    # draw_cancergrade_mask(title, scc, org_size, output_size, SAVE_DIR)

    # title = wsi_name + "_mask_level0" + str(level) + "_grade04"
    # draw_cancergrade_mask(title, others, org_size, output_size, SAVE_DIR)

    # if len(etc) > 0:
    #     title = SAVE_DIR + wsi_name + "_mask_etc.txt"
    #     with open(title, "w", encoding="utf-8") as f:
    #         for ele in etc:
    #             f.write(f"{ele[0]}\n")
    #  # =================================== #
    wsi.close()


def make_cancergrade_mask(
    pre_gt_dir: str, bg_mask_path: str, save_dir: str, level: int=5, wsi_name: str=None
):
    bg_mask = cv2.imread(bg_mask_path, cv2.IMREAD_GRAYSCALE)
    bg_mask = cv2.bitwise_not(bg_mask)
    nolabel_mask = np.zeros(bg_mask.shape, dtype=np.uint8)

    grade_paths = natsorted(glob.glob(f"{pre_gt_dir}{wsi_name}_mask_level0{str(level)}_*.tif"))
    # 既にnolabelが含まれている場合は除去
    grade_paths = [path for path in grade_paths if "nolabel" not in path]


    for grade_path in grade_paths:
        grade_mask = cv2.imread(grade_path, cv2.IMREAD_GRAYSCALE)
        filename = get_filename(grade_path)

        # bg_maskと各gradeのpre_gtの論理積
        new_gt = cv2.bitwise_and(grade_mask, bg_mask)
        cv2.imwrite(save_dir + filename + ".tif", new_gt)
        nolabel_mask = cv2.bitwise_or(nolabel_mask, new_gt)

    nolabel_mask = cv2.bitwise_xor(nolabel_mask, bg_mask)
    cv2.imwrite(
        save_dir + wsi_name + "_mask_level0" + str(level) + "_nolabel.tif",
        nolabel_mask,
    )


if __name__ == "__main__":
    PARENT_DIR = "/mnt/ssdwdc/ResearchData/chemotherapy/202110_chemotherapy/"

    ORIGIN_DIR = PARENT_DIR + "origin/"
    GT_TMP_DIR = PARENT_DIR + "mask_cancergrade/"
    BG_MASK_DIR = PARENT_DIR + "mask_bg/"
    SAVE_DIR = PARENT_DIR + "mask_cancergrade/"

    labels = ['Tumor range', 'Residual tumor']
    level = 5

    WSIs = natsorted(glob.glob(ORIGIN_DIR + "*.ndpi"))
    for wsi_path in WSIs:
        wsi_name = get_filename(wsi_path)
        xml_path = ORIGIN_DIR + wsi_name + ".xml"

        annot_dict = load_xml_file(
            xml_path, labels=labels, log_path=None
        )
        make_pre_cancergrade(
            wsi_path, annot_dict, GT_TMP_DIR, level=level, wsi_name=wsi_name
        )

        bg_mask_path = f"{BG_MASK_DIR}{wsi_name}_mask_level0{level}_bg.tif"
        make_cancergrade_mask(
            GT_TMP_DIR, bg_mask_path, SAVE_DIR, level=level, wsi_name=wsi_name
        )
