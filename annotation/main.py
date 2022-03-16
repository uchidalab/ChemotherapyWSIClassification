# -*- coding: utf-8 -*-

import os
from natsort import natsorted
import glob
from make_bg_mask import make_bg_mask, make_small_wsi
from make_gt_from_xml import load_xml_file, make_pre_cancergrade, make_cancergrade_mask
from make_overlaid_mask import make_overlaid
from util import get_filename, chemotherapy_label_dict


def main():
    PARENT_DIR = "/mnt/ssdwdc/ResearchData/chemotherapy/202112_chemotherapy/"

    MODE = {
        "cleanup_dir": False,
        "make_bg_mask": True,
        "make_cancergrade_mask": True,
        "make_overlaid_mask": True,
    }

    ORIGIN_DIR = PARENT_DIR + "origin/"
    SMALLFOLDER_DIR = PARENT_DIR + "smallfolder/"
    BG_MASK_DIR = PARENT_DIR + "mask_bg/"
    CANCERGRADE_DIR = PARENT_DIR + "mask_cancergrade/"
    OVERLAID_DIR = PARENT_DIR + "overlaid_[0, 1, 2]/"

    LOG_FILE = PARENT_DIR + "error_log.txt"

    DOWN_LEVEL = 5  # 縮小サイズ
    LABELS = ['Tumor bed', 'Residual tumor']
    LABEL_DICT = chemotherapy_label_dict
    OVERLAID_TH = 250
    IS_NOLABEL_NORMAL = True
    KERNEL_SIZE = 5
    MASK_BG_ITERATIONS=5


    # # === cleanup dir === #
    if MODE["cleanup_dir"]:
        if os.path.isdir(SMALLFOLDER_DIR) is False:
            os.mkdir(SMALLFOLDER_DIR)
        if os.path.isdir(BG_MASK_DIR) is False:
            os.mkdir(BG_MASK_DIR)
        if os.path.isdir(CANCERGRADE_DIR) is False:
            os.mkdir(CANCERGRADE_DIR)
        if os.path.isdir(OVERLAID_DIR) is False:
            os.mkdir(OVERLAID_DIR)
            os.mkdir(OVERLAID_DIR + "rgb/")
            os.mkdir(OVERLAID_DIR + "gray/")

    # === make_bg_mask & small_image === #
    if MODE["make_bg_mask"]:
        print("== make bg_mask ==")
        WSIs = natsorted(glob.glob(ORIGIN_DIR + "*.ndpi"))
        for wsi_path in WSIs:
            wsi_name = get_filename(wsi_path)
            make_small_wsi(wsi_path, wsi_name, SMALLFOLDER_DIR, level=DOWN_LEVEL)
        make_bg_mask(SMALLFOLDER_DIR, BG_MASK_DIR, DOWN_LEVEL, kernel_size=KERNEL_SIZE, iterations=MASK_BG_ITERATIONS)

    # === make_gt_from_xml === #
    if MODE["make_cancergrade_mask"]:
        print("== make gt_from_xml ==")

        WSIs = natsorted(glob.glob(ORIGIN_DIR + "*.ndpi"))
        for wsi_path in WSIs:
            wsi_name = get_filename(wsi_path)
            xml_path = ORIGIN_DIR + wsi_name + ".xml"

            # xmlファイルがない場合
            if os.path.exists(xml_path) is False:
                print(f"xml: {xml_path} does not exists")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"xml-error: {wsi_name}'s xml does not exists\n")
                continue

            annot_dict = load_xml_file(
                xml_path, labels=LABELS, label_dict=LABEL_DICT, log_path=LOG_FILE
            )
            make_pre_cancergrade(
                wsi_path,
                annot_dict,
                CANCERGRADE_DIR,
                level=DOWN_LEVEL,
                wsi_name=wsi_name,
            )

            bg_mask_path = f"{BG_MASK_DIR}{wsi_name}_mask_level0{DOWN_LEVEL}_bg.tif"
            make_cancergrade_mask(
                CANCERGRADE_DIR,
                bg_mask_path,
                CANCERGRADE_DIR,
                level=DOWN_LEVEL,
                wsi_name=wsi_name,
            )

    # === make_overlaid_mask === #
    if MODE["make_overlaid_mask"]:
        print("== make overlaid_mask ==")
        make_overlaid(
            CANCERGRADE_DIR,
            BG_MASK_DIR,
            OVERLAID_DIR,
            th=OVERLAID_TH,
            labels=LABELS,
            is_nolabel_normal=IS_NOLABEL_NORMAL
        )


if __name__ == "__main__":
    main()
