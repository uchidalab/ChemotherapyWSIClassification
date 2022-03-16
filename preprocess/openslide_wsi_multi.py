import os
import openslide
import pathlib

import numpy as np
from PIL import Image
import cv2
from scipy import stats
from natsort import natsorted


# TODO: multi-scale用のbb_to_patch, patch_to_imgを作成

class OpenSlideWSI(openslide.OpenSlide):

    def __init__(self, filename, bg_mask_dir=None, semantic_mask_dir=None):
        super().__init__(filename)
        p_filename = pathlib.Path(filename)
        self.wsi_name = str(p_filename.stem)
        self.wsi_obj_format = '{wsi_name}_{obj_idx:03d}'.format

        if bg_mask_dir is not None:
            self.bg_mask_dir = bg_mask_dir
            self.filename_bg_mask = self.bg_mask_dir \
                + self.wsi_name + "_mask_level05_bg.tif"
        if semantic_mask_dir is not None:
            self.semantic_mask_dir = semantic_mask_dir
            self.filename_semantic_mask = self.semantic_mask_dir \
                + self.wsi_name + "_overlaid.tif"

    # 条件を満たさなければNoneを返す
    def _get_output_dir(self, s_p, output_main_dir: str, obj_name: str, th: int = 1):
        output_dir = None
        s_p_mode = stats.mode(
            s_p, axis=None
        )  # s_p_mode[0]:s_pの最頻値, s_p_mode[1]:s_pの最頻値の個数
        # 背景領域が多いパッチは除外
        if int(s_p_mode[0]) != 255:
            # semanticパッチのクラス最頻値のピクセル数の割合と閾値を比較
            if float(s_p_mode[1] / (s_p.shape[0] * s_p.shape[1])) >= th:
                output_dir = f"{output_main_dir}{int(s_p_mode[0])}/{obj_name}/"
                os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    # 条件を満たさなければNoneを返す
    def _get_output_multi_dir(self, s_p, output_main_dir: str, obj_name: str, levels: list, th: int = 1):
        output_dir_0, output_dir_1, output_dir_2 = None, None, None
        s_p_mode = stats.mode(
            s_p, axis=None
        )  # s_p_mode[0]:s_pの最頻値, s_p_mode[1]:s_pの最頻値の個数
        # 背景領域が多いパッチは除外
        if int(s_p_mode[0]) != 255:
            # semanticパッチのクラス最頻値のピクセル数の割合と閾値を比較
            if float(s_p_mode[1] / (s_p.shape[0] * s_p.shape[1])) >= th:
                output_dir_0 = f"{output_main_dir}{int(s_p_mode[0])}/{obj_name}/{levels[0]}/"
                os.makedirs(output_dir_0) if os.path.isdir(output_dir_0) is False else None
                output_dir_1 = f"{output_main_dir}{int(s_p_mode[0])}/{obj_name}/{levels[1]}/"
                os.makedirs(output_dir_1) if os.path.isdir(output_dir_1) is False else None
                output_dir_2 = f"{output_main_dir}{int(s_p_mode[0])}/{obj_name}/{levels[2]}/"
                os.makedirs(output_dir_2) if os.path.isdir(output_dir_2) is False else None
        return output_dir_0, output_dir_1, output_dir_2

    def _make_output_dir(self, output_main_dir: str, obj_name: str):
        output_dir = f"{output_main_dir}/{obj_name}/"
        os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    def _make_output_multi_dir(self, output_main_dir: str, obj_name: str, levels: list):
        output_dir_0 = f"{output_main_dir}/{obj_name}/{levels[0]}/"
        os.makedirs(output_dir_0) if os.path.isdir(output_dir_0) is False else None
        output_dir_1 = f"{output_main_dir}/{obj_name}/{levels[1]}/"
        os.makedirs(output_dir_1) if os.path.isdir(output_dir_1) is False else None
        output_dir_2 = f"{output_main_dir}/{obj_name}/{levels[2]}/"
        os.makedirs(output_dir_2) if os.path.isdir(output_dir_2) is False else None
        return output_dir_0, output_dir_1, output_dir_2

    def _getBoundingBox(self, test_dir=None):
        bg_mask = cv2.imread(self.filename_bg_mask, cv2.IMREAD_GRAYSCALE)
        bg_mask_inv = np.zeros((bg_mask.shape), dtype=np.uint8)
        bg_mask_inv[bg_mask == 0] = 255

        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask_inv)

        bb_list = []
        obj_idx = 0
        for i in range(1, nlabels):
            x, y, w, h, obj_size = stats[i]
            wsi_obj_name = self.wsi_obj_format(wsi_name=self.wsi_name, obj_idx=obj_idx)

            if obj_size >= 10000:
                cv2.rectangle(bg_mask_inv, (x, y), (x + w, y + h), 255, cv2.LINE_4)
                stat = stats[i]
                bb_list.append({'x': stat[0], 'y': stat[1], 'w': stat[2],
                                'h': stat[3], 'name': wsi_obj_name})
                obj_idx += 1
            else:
                print(f"[Exclude] {wsi_obj_name} (size: {obj_size})")
                bg_mask_inv[labels == i] = 0

        if (test_dir is not None) and (obj_idx >= 1):
            cv2.imwrite(test_dir + wsi_obj_name + ".png", bg_mask_inv)

        return bb_list

    def bb_to_patch(
        self,
        default_level,
        level,
        size,
        stride,
        bb,
        output_main_dir,
        contours_th=0.5,
    ):  # size=(width, height)
        """
        default_level: bg_mask, semantic_maskの倍率(基本は5に設定)
        """
        assert isinstance(bb, dict), "bb(bounding-box) must be dict type"
        obj_name = bb['name']

        # bbのx, yをlevel0用に変換（bbはbg_mask_level05における座標のため）
        bx_wsi = bb['x'] * (2 ** default_level)  # bounding-boxの左上x座標(level0)
        by_wsi = bb['y'] * (2 ** default_level)  # bounding-boxの左上y座標(level0)

        # bbのw, hを特定のlevel用に変換
        bw_wsi = bb['w'] * (2 ** (default_level - level))  # bounding-boxの横幅(level)
        bh_wsi = bb['h'] * (2 ** (default_level - level))  # bounding-boxの縦幅(level)

        row_max = int((bw_wsi - size[0]) / stride + 1)
        column_max = int((bh_wsi - size[1]) / stride + 1)

        # 細胞領域のマスク画像，背景領域のマスク画像の１ピクセルが特定のレベルのWSIの何ピクセルに相当するか計算
        # 細胞領域のマスク画像，背景領域のマスク画像(default_level)のstride, width, height(単位はpixel)が
        #   特定のレベル(level)において何pixelに該当するか算出
        stride_rate = stride / 2 ** (default_level - level)
        width_rate = size[0] / 2 ** (default_level - level)
        height_rate = size[1] / 2 ** (default_level - level)

        assert self.filename_semantic_mask is not None, "Should set filename_semantic_mask"
        semantic_mask = Image.open(self.filename_semantic_mask)
        semantic_mask_np = np.array(semantic_mask)
        assert self.filename_bg_mask is not None, "Should set filename_bg_mask"
        bg_mask_np = np.array(Image.open(self.filename_bg_mask))

        cnt = 0
        for column in range(column_max):
            for row in range(row_max):
                i = int(bx_wsi + (row * stride * (2 ** level)))
                j = int(by_wsi + (column * stride * (2 ** level)))

                mask_base_idx = {'row': int(bb['x'] + (row * stride_rate)),
                                 'col': int(bb['y'] + (column * stride_rate))}

                # width_rate×height_rateの領域(背景領域のマスク画像)の画素値が0の画素数で比較
                if (
                    len(
                        np.where(
                            bg_mask_np[
                                mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                                mask_base_idx['row']:int(mask_base_idx['row'] + width_rate),
                            ]
                            == 0
                        )[0]
                    )
                    >= contours_th * height_rate * width_rate
                ):
                    # width_rate×height_rateの領域(semanticマスク)の画素値が255以外(背景以外)の画素数で比較
                    if (
                        len(
                            np.where(
                                semantic_mask_np[
                                    mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                                    mask_base_idx['row']:int(mask_base_idx['row'] + width_rate),
                                ]
                                != 255
                            )[0]
                        )
                        >= contours_th * height_rate * width_rate
                    ):

                        s_p = semantic_mask_np[
                            mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                            mask_base_idx['row']:int(mask_base_idx['row'] + width_rate)
                        ]

                        output_dir = self._get_output_dir(
                            s_p, output_main_dir, obj_name)

                        # if output_dir is not None:
                        #     self.read_region((i, j), level, size).save(
                        #         output_dir
                        #         + str(level)
                        #         + "_"
                        #         + str(cnt).zfill(10)
                        #         + ".png"
                        #     )
                        #     cnt = cnt + 1

    def bb_to_patch_multi(
        self,
        default_level: int,
        levels: list,
        size: tuple,
        stride: int,
        bb: tuple,
        output_main_dir: str,
        contours_th: float = 0.5,
    ):  # size=(width, height)
        assert isinstance(bb, dict), "bb(bounding-box) must be dict type"
        obj_name = bb['name']

        # bbのx, yを最大倍率用に変換（bbはbg_mask_level05における座標のため）
        bx_wsi = bb['x'] * (2 ** default_level)  # bounding-boxの左上x座標(最大倍率)
        by_wsi = bb['y'] * (2 ** default_level)  # bounding-boxの左上y座標(最大倍率)

        # bbのw, hを特定のlevels[0]用に変換
        bw_wsi_0 = bb['w'] * (2 ** (default_level - levels[0]))  # bounding-boxの横幅(levels[0])
        bh_wsi_0 = bb['h'] * (2 ** (default_level - levels[0]))  # bounding-boxの縦幅(levels[0])

        row_max = int((bw_wsi_0 - size[0]) / stride + 1)
        column_max = int((bh_wsi_0 - size[1]) / stride + 1)

        # 細胞領域のマスク画像，背景領域のマスク画像の１ピクセルが特定のレベルのWSIの何ピクセルに相当するか計算
        stride_rate_0 = stride / 2 ** (default_level - levels[0])
        width_rate_0 = size[0] / 2 ** (default_level - levels[0])
        height_rate_0 = size[1] / 2 ** (default_level - levels[0])

        assert self.filename_semantic_mask is not None, "Should set filename_semantic_mask"
        semantic_mask = Image.open(self.filename_semantic_mask)
        semantic_mask_np = np.array(semantic_mask)
        assert self.filename_bg_mask is not None, "Should set filename_bg_mask"
        bg_mask_np = np.array(Image.open(self.filename_bg_mask))

        cnt = 0
        for column in range(column_max):
            for row in range(row_max):
                i0 = int(bx_wsi + (row * stride * (2 ** levels[0])))
                j0 = int(by_wsi + (column * stride * (2 ** levels[0])))

                mask_base_idx = {'row': int(bb['x'] + (row * stride_rate_0)),
                                 'col': int(bb['y'] + (column * stride_rate_0))}

                # width_rate×height_rateの領域(背景領域のマスク画像)の画素値が0の画素数で比較
                if (
                    len(
                        np.where(
                            bg_mask_np[
                                mask_base_idx['col']:int(mask_base_idx['col'] + height_rate_0),
                                mask_base_idx['row']:int(mask_base_idx['row'] + width_rate_0),
                            ]
                            == 0
                        )[0]
                    )
                    >= contours_th * height_rate_0 * width_rate_0
                ):
                    # width_rate×height_rateの領域(semanticマスク)の画素値が255以外(背景以外)の画素数で比較
                    if (
                        len(
                            np.where(
                                semantic_mask_np[
                                    mask_base_idx['col']:int(mask_base_idx['col'] + height_rate_0),
                                    mask_base_idx['row']:int(mask_base_idx['row'] + width_rate_0),
                                ]
                                != 255
                            )[0]
                        )
                        >= contours_th * height_rate_0 * width_rate_0
                    ):

                        s_p = semantic_mask_np[
                            mask_base_idx['col']:int(mask_base_idx['col'] + height_rate_0),
                            mask_base_idx['row']:int(mask_base_idx['row'] + width_rate_0)
                        ]

                        output_dir_0, output_dir_1, output_dir_2 = self._get_output_multi_dir(
                            s_p, output_main_dir, obj_name, levels=LEVELS)

                        if output_dir_0 is not None:
                            if len(levels) == 3:
                                # level[1]の左上座標(i1, j1)算出に使用する左上座標(i0, j0)からの移動量
                                mv_left1 = ((2 ** (levels[1] - levels[0]) - 1) / 2) * size[0]
                                mv_up1 = ((2 ** (levels[1] - levels[0]) - 1) / 2) * size[1]

                                # level[2]の左上座標(i2, j2)算出に使用する左上座標(i0, j0)からの移動量
                                mv_left2 = ((2 ** (levels[2] - levels[0]) - 1) / 2) * size[0]
                                mv_up2 = ((2 ** (levels[2] - levels[0]) - 1) / 2) * size[1]

                                # level[1]のパッチの左上座標(座標は最大倍率における画素位置)
                                i1 = int(i0 - mv_left1)
                                j1 = int(j0 - mv_up1)

                                # level[2]のパッチの左上座標(座標は最大倍率における画素位置)
                                i2 = int(i0 - mv_left2)
                                j2 = int(j0 - mv_up2)

                                if i1 < 0 or j1 < 0:
                                    print(f"LEVEL{LEVELS[1]} cannot crop, because it will be sticked out from WSI\n (i1: {i1}, j1: {j1})")
                                elif i2 < 0 or j2 < 0:
                                    print(f"LEVEL{LEVELS[2]} cannot crop, because it will be sticked out from WSI\n (i1: {i2}, j1: {j2})")
                                else:
                                    self.read_region((i0, j0), levels[0], size).save(
                                        output_dir_0
                                        + str(levels[0])
                                        + "_"
                                        + str(cnt).zfill(10)
                                        + ".png"
                                    )
                                    self.read_region((i1, j1), levels[1], size).save(
                                        output_dir_1
                                        + str(levels[1])
                                        + "_"
                                        + str(cnt).zfill(10)
                                        + ".png"
                                    )
                                    self.read_region((i2, j2), levels[2], size).save(
                                        output_dir_2
                                        + str(levels[2])
                                        + "_"
                                        + str(cnt).zfill(10)
                                        + ".png"
                                    )
                                    cnt = cnt + 1

    # split a full-size image to patch images
    # this patch is used for making prediction-map

    def image_to_patch(
        self,
        level: int,
        size: tuple,
        stride: int,
        output_main_dir: str,
        obj_name: str,
        cnt: int = 0,
    ):
        """
        single-scaleのsegmentation-map用

        size: (width, height)
        """

        width = self.level_dimensions[level][0]
        height = self.level_dimensions[level][1]
        row_max = int((width - size[0]) / stride + 1)
        column_max = int((height - size[1]) / stride + 1)

        # bg_maskとsemantic_maskを入力に与えなかった場合
        for column in range(column_max):
            for row in range(row_max):
                i = int(row * stride * (2 ** level))
                j = int(column * stride * (2 ** level))

                output_dir = self._make_output_dir(output_main_dir, obj_name)

                self.read_region((i, j), level, size).save(
                    output_dir
                    + str(level)
                    + "_"
                    + str(cnt).zfill(10)
                    + ".png"
                )

                cnt = cnt + 1
        return cnt

    def image_to_patch_multi(
        self,
        levels: list,
        size: tuple,
        stride: int,
        output_main_dir: str,
        obj_name: str,
        cnt: int = 0,
    ):
        """
        multi-scaleのsegmentation-map用

        size: (width, height)
        """

        width = self.level_dimensions[levels[0]][0]
        height = self.level_dimensions[levels[0]][1]
        row_max = int((width - size[0]) / stride + 1)
        column_max = int((height - size[1]) / stride + 1)

        # bg_maskとsemantic_maskを入力に与えなかった場合
        for column in range(column_max):
            for row in range(row_max):
                i = int(row * stride * (2 ** levels[0]))
                j = int(column * stride * (2 ** levels[0]))

                output_dir_0, output_dir_1, output_dir_2 = \
                    self._make_output_multi_dir(output_main_dir, obj_name, levels)

                self.read_region((i, j), levels[0], size).save(
                    output_dir_0
                    + str(levels[0])
                    + "_"
                    + str(cnt).zfill(10)
                    + ".png"
                )
                if len(levels) == 3:
                    i1 = int(
                        i
                        + size[0]
                        * 2 ** levels[1]
                        * (1.0 / 2 ** (1 + levels[1] - levels[0]) - 1.0 / 2)
                    )
                    j1 = int(
                        j
                        + size[1]
                        * 2 ** levels[1]
                        * (1.0 / 2 ** (1 + levels[1] - levels[0]) - 1.0 / 2)
                    )
                    i2 = int(
                        i
                        + size[0]
                        * 2 ** levels[2]
                        * (1.0 / 2 ** (1 + levels[2] - levels[0]) - 1.0 / 2)
                    )
                    j2 = int(
                        j
                        + size[1]
                        * 2 ** levels[2]
                        * (1.0 / 2 ** (1 + levels[2] - levels[0]) - 1.0 / 2)
                    )
                    self.read_region((i1, j1), levels[1], size).save(
                        output_dir_1
                        + str(levels[1])
                        + "_"
                        + str(cnt).zfill(10)
                        + ".png"
                    )
                    self.read_region((i2, j2), levels[2], size).save(
                        output_dir_2
                        + str(levels[2])
                        + "_"
                        + str(cnt).zfill(10)
                        + ".png"
                    )
                cnt = cnt + 1

    # merge patch images to a full size image
    def patch_to_image(
        self,
        resized_size,
        level,
        size,
        stride,
        input_dir,
        output_dir,
        output_name,
        suffix=None,
        cnt=0,
    ):

        width = self.level_dimensions[level][0]
        height = self.level_dimensions[level][1]
        row_max = int((width - size[0]) / stride + 1)
        column_max = int((height - size[1]) / stride + 1)

        canvas = Image.new(
            "RGB",
            (resized_size[0] * row_max, resized_size[1] * column_max),
            (255, 255, 255),
        )

        for column in range(column_max):
            for row in range(row_max):
                if suffix is None:
                    img = Image.open(
                        input_dir + str(level) + "_" + str(cnt).zfill(10) + ".png", "r"
                    ).resize((resized_size[0], resized_size[1]))
                else:
                    img = Image.open(
                        input_dir + str(level) + "_" + str(cnt).zfill(10) + str(suffix) + ".png", "r"
                    ).resize((resized_size[0], resized_size[1]))
                canvas.paste(img, (row * resized_size[0], column * resized_size[1]))
                cnt = cnt + 1
        if suffix is None:
            canvas.save(output_dir + output_name + ".png", "PNG", quality=100)
        else:
            canvas.save(output_dir + output_name + str(suffix) + ".png", "PNG", quality=100)

        return cnt


def main(
    PARENT_DIR: str = "/mnt/ssdsam/chemotherapy_strage/mnt1/",
    DEFAULT_LEVEL: int = 5,
    LEVEL: int = 0,
    SIZE: tuple = (256, 256),
    STRIDE: int = 256,
    CONTOURS_TH: int = 1,
    CLASSES: list = [0, 1, 2],
):
    p_parent_dir = pathlib.Path(PARENT_DIR)
    output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt2_LEV{LEVEL}/")

    wsi_list = natsorted([wsi_path for wsi_path in (p_parent_dir / "origin/").glob("*.ndpi")])

    skip_list = [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "0/"))]
    skip_list += [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "1/"))]
    skip_list += [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "2/"))]
    skip_list = list(set(skip_list))
    print(len(skip_list))

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)
        bg_mask_dir = PARENT_DIR + "mask_bg/"
        semantic_mask_dir = PARENT_DIR + \
            f"mask_cancergrade_gray/overlaid_{CLASSES}/"

        tmp_skip_list = [s for s in skip_list if s in wsi_path]
        if len(tmp_skip_list) > 0:
            print(f"skip: {wsi_path}")
            continue

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=bg_mask_dir,
            semantic_mask_dir=semantic_mask_dir)

        print("==== {} ====".format(wsi.wsi_name))

        bb_list = wsi._getBoundingBox()

        for bb in bb_list:
            print(bb['name'])

            wsi.bb_to_patch(
                DEFAULT_LEVEL,
                LEVEL,
                SIZE,
                STRIDE,
                bb,
                output_main_dir,
                CONTOURS_TH,
            )


def main_multi(
    PARENT_DIR: str = "/mnt/ssdsam/chemotherapy_strage/mnt1/",
    DEFAULT_LEVEL: int = 5,
    LEVELS: list = [0, 1, 2],
    SIZE: tuple = (256, 256),
    STRIDE: int = 256,
    CONTOURS_TH: int = 1,
    CLASSES: list = [0, 1, 2],
):
    """
    multi-scale用のパッチ切り取り(３つの倍率を想定)
    """

    p_parent_dir = pathlib.Path(PARENT_DIR)
    # output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt2_LEV{LEVELS}/")
    output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt2_LEV{''.join(map(str, LEVELS))}/")

    wsi_list = natsorted([wsi_path for wsi_path in (p_parent_dir / "origin/").glob("*.ndpi")])

    skip_list = [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "0/"))]
    skip_list += [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "1/"))]
    skip_list += [mnt2_indir.replace("_000", "") for mnt2_indir in os.listdir(str(output_main_dir + "2/"))]
    skip_list = list(set(skip_list))
    print(len(skip_list))

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)
        bg_mask_dir = PARENT_DIR + "mask_bg/"
        semantic_mask_dir = PARENT_DIR + \
            f"mask_cancergrade_gray/overlaid_{CLASSES}/"

        tmp_skip_list = [s for s in skip_list if s in wsi_path]
        if len(tmp_skip_list) > 0:
            print(f"skip: {wsi_path}")
            continue

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=bg_mask_dir,
            semantic_mask_dir=semantic_mask_dir)

        print("==== {} ====".format(wsi.wsi_name))

        bb_list = wsi._getBoundingBox()

        for bb in bb_list:
            print(bb['name'])

            wsi.bb_to_patch_multi(
                DEFAULT_LEVEL,
                LEVELS,
                SIZE,
                STRIDE,
                bb,
                output_main_dir,
                CONTOURS_TH,
            )

# 予測画像用のパッチ切り取り


def main_for_predmap(
    PARENT_DIR: str,
    LEVEL: int = 0,
    SIZE: tuple = (256, 256),
    STRIDE: int = 256,
):
    p_parent_dir = pathlib.Path(PARENT_DIR)
    output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt3_LEV{LEVEL}/")

    wsi_list = natsorted([wsi_path for wsi_path in (p_parent_dir / "origin/").glob("*.ndpi")])

    # mnt3に存在するWSIはスキップ
    skip_list = [mnt2_indir for mnt2_indir in os.listdir(str(output_main_dir))]
    skip_list = list(set(skip_list))

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)

        tmp_skip_list = [s for s in skip_list if s in wsi_path]
        if len(tmp_skip_list) > 0:
            print(f"skip: {wsi_path}")
            continue

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=None,
            semantic_mask_dir=None)

        print("==== {} ====".format(wsi.wsi_name))
        wsi.image_to_patch(LEVEL, SIZE, STRIDE, output_main_dir, wsi.wsi_name)


def main_for_predmap_multi(
    PARENT_DIR: str,
    LEVELS: list,
    SIZE: tuple = (256, 256),
    STRIDE: int = 256,
):
    p_parent_dir = pathlib.Path(PARENT_DIR)

    # output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt3_LEV{LEVELS}/")
    # output_main_dir = PARENT_DIR.replace("mnt1/", f"mnt3_LEV{''.join(map(str, LEVELS))}/")

    output_main_dir = PARENT_DIR.replace("ssdsam/", f"ssdwdc/")
    output_main_dir = output_main_dir.replace("mnt1/", f"mnt3_LEV{''.join(map(str, LEVELS))}/")

    wsi_list = natsorted([wsi_path for wsi_path in (p_parent_dir / "origin/").glob("*.ndpi")])

    # mnt3に存在するWSIはスキップ
    skip_list = [mnt2_indir for mnt2_indir in os.listdir(str(output_main_dir))]
    skip_list = list(set(skip_list))

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)

        tmp_skip_list = [s for s in skip_list if s in wsi_path]
        if len(tmp_skip_list) > 0:
            print(f"skip: {wsi_path}")
            continue

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=None,
            semantic_mask_dir=None)

        print("==== {} ====".format(wsi.wsi_name))
        wsi.image_to_patch_multi(LEVELS, SIZE, STRIDE, output_main_dir, wsi.wsi_name)


if __name__ == "__main__":
    PARENT_DIR = "/mnt/ssdsam/chemotherapy_strage/mnt1/"
    LEVELS = [0, 1, 2]
    CLASSES = [0, 1, 2]

    # main(
    #     PARENT_DIR=PARENT_DIR,
    #     LEVEL=LEVEL,
    #     CLASSES=CLASSES
    # )

    # main_multi(
    #     PARENT_DIR=PARENT_DIR,
    #     LEVELS=LEVELS,
    #     CLASSES=CLASSES
    # )
    # main(
    #     PARENT_DIR=PARENT_DIR,
    #     LEVEL=0,
    #     CLASSES=CLASSES
    # )
    # main(
    #     PARENT_DIR=PARENT_DIR,
    #     LEVEL=1,
    #     CLASSES=CLASSES
    # )

    # main_for_predmap(PARENT_DIR=PARENT_DIR, LEVEL=LEVEL)
    main_for_predmap_multi(PARENT_DIR=PARENT_DIR, LEVELS=LEVELS)
