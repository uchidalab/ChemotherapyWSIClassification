import os
import cv2
import glob
import numpy as np
from natsort import natsorted
from openslide import OpenSlide


# make small-wsi
def make_small_wsi(wsi_path, wsi_name, save_dir, level=5):
    wsi = OpenSlide(wsi_path)
    print("org: {}".format(wsi.dimensions))

    # get resized-image size
    resized_width = wsi.level_dimensions[level][0]
    resized_height = wsi.level_dimensions[level][1]
    output_size = (resized_width, resized_height)
    print("size:{}".format(output_size))

    # small_wsi = wsi.get_thumbnail(output_size)
    small_wsi = wsi.read_region((0, 0), 5, output_size)
    small_wsi.save(save_dir + wsi_name + "_small_level0" + str(level) + ".tif")
    wsi.close()


def remove_objects(img, lower_size):
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[:, -1]
    _img = np.zeros(img.shape, dtype=np.uint8)

    # label=0はbackgroundのため,1から開始
    for i in range(1, nlabels):
        # 小さいオブジェクトの除去
        if lower_size < sizes[i]:
            _img[labels == i] = 255
    return _img


# make bg_mask
def make_bg_mask(src_dir, save_dir, down_level, kernel_size=3, iterations=2):

    small_images = sorted(glob.glob(src_dir + "*.tif"))

    for small_img_path in small_images:
        fn = os.path.splitext(os.path.basename(small_img_path))[0]
        wsi_name = fn.replace("_small_level05", "")
        print(wsi_name)

        img = cv2.imread(small_img_path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 修正前
        lower = np.array([130, 20, 120])
        upper = np.array([180, 255, 255])

        # # 修正後
        # lower = np.array([130, 40, 120])
        # upper = np.array([180, 255, 255])

        img_mask = cv2.inRange(hsv, lower, upper)
        img = cv2.bitwise_and(img, img, mask=img_mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

        # Closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # kernel size は奇数！
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

        _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

        # remove small objects
        img = remove_objects(img, lower_size=1000)

        img = cv2.bitwise_not(img)
        # img = 255 - img
        cv2.imwrite(
            save_dir + wsi_name + "_mask_level0" + str(down_level) + "_bg.tif", img
        )


if __name__ == "__main__":
    down_level = 5  # 縮小画像用
    wsi_type = "bif"
    wsi_fold = "11"

    parent_dir = (
        "/home/kengoaraki/ResearchData/CervicalCancer/20191030_Uterine_cervix_cancer/"
    )
    save_smallfolder_dir = parent_dir + "MF00" + wsi_fold + "/smallfolder/"
    save_bg_dir = parent_dir + "MF00" + wsi_fold + "/mask_bg(revised)/"

    # ===============================#
    #        make small wsi         #
    # ===============================#
    src_dir = parent_dir + "MF00" + wsi_fold + "/origin/"
    files = natsorted(os.listdir(src_dir))
    files_dir = [f for f in files if os.path.isdir(os.path.join(src_dir, f))]

    for i in range(len(files_dir)):
        dir_name = files_dir[i]
        print(dir_name)

        WSIs = sorted(glob.glob(src_dir + dir_name + "/*." + wsi_type))
        for wsi_path in WSIs:
            # src_name = return_filename(wsi_path)
            src_name = os.path.splitext(os.path.basename(wsi_path))[0]
            print(src_name)

            wsi_name = wsi_fold + "_" + dir_name + "_" + src_name
            make_small_wsi(wsi_path, wsi_name, save_smallfolder_dir, level=down_level)

    # ===============================#
    #        make bg mask           #
    # ===============================#
    make_bg_mask(save_smallfolder_dir, save_bg_dir, down_level)
