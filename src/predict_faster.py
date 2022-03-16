import logging
import os
import sys
import yaml
import joblib
import glob
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from openslide import OpenSlide

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import WSI
from src.util import fix_seed
from src.model import build_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class makePredmap(object):
    def __init__(self, wsi_name, classes, level, wsi_dir, overlaid_mask_dir):
        self.wsi_name = wsi_name
        self.classes = classes

        self.wsi_dir = wsi_dir
        self.wsi_path = f"{self.wsi_dir}{self.wsi_name}.ndpi"

        self.overlaid_mask_dir = overlaid_mask_dir

        self.default_level = 5
        self.level = level
        self.length = 256
        self.resized_size = (
            int(self.length / 2 ** (self.default_level - self.level)),
            int(self.length / 2 ** (self.default_level - self.level))
        )
        self.size = (self.length, self.length)
        self.stride = 256

    def get_wsi_name(self):
        return self.wsi_name

    def num_to_color(self, num):
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

    # 予測結果からセグメンテーション画像を生成
    def preds_to_image(
        self,
        preds: list,
        output_dir: str,
        output_name: str,
        cnt=0,
    ):
        wsi = OpenSlide(self.wsi_path)

        width = wsi.level_dimensions[self.level][0]
        height = wsi.level_dimensions[self.level][1]
        row_max = int((width - self.size[0]) / self.stride + 1)
        column_max = int((height - self.size[1]) / self.stride + 1)

        canvas_shape = (
            self.resized_size[1] * column_max,
            self.resized_size[0] * row_max, 3)
        canvas_nd = np.full(canvas_shape, 255, dtype=np.uint8)

        for column in range(column_max):
            for row in range(row_max):
                y = preds[cnt].argmax(dim=0).numpy().copy()
                y_color = self.num_to_color(y)

                canvas_nd[
                    column * self.resized_size[1]:(column + 1) * self.resized_size[1],
                    row * self.resized_size[0]:(row + 1) * self.resized_size[0], :] = y_color
                cnt = cnt + 1
        canvas = Image.fromarray(canvas_nd)
        canvas.save(output_dir + output_name + ".png", "PNG", quality=100)

    # 背景&対象外領域をマスク
    def make_black_mask(self, input_dir, output_dir, suffix=None):
        if suffix is None:
            filename = self.wsi_name
        else:
            filename = self.wsi_name + suffix

        image = Image.open(
            input_dir + filename + ".png"
        )
        image_gt = Image.open(self.overlaid_mask_dir + self.wsi_name + "_overlaid.tif")

        WIDTH = image.size[0]
        HEIGHT = image.size[1]

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if image_gt.getpixel((x, y)) == (0, 0, 0):
                    image.putpixel((x, y), (0, 0, 0))
                elif image_gt.getpixel((x, y)) == (255, 255, 255):
                    image.putpixel((x, y), (255, 255, 255))

        image.save(
            output_dir + filename + ".png",
            "PNG",
            quality=100,
            optimize=True,
        )


def main():
    fix_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='../config/config_src_LEV0.yaml')
    parser.add_argument('--main_dir', default='/mnt/ssdsam/chemotherapy_strage/')

    args = parser.parse_args()
    config_path = args.config_path
    MAIN_DIR = args.main_dir

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    # ========================================================== #
    WSI_DIR = MAIN_DIR + "mnt1/origin/"
    MASK_DIR = MAIN_DIR + f"mnt1/mask_cancergrade/overlaid_{config['main']['classes']}/"
    PATCH_DIR = MAIN_DIR + f"mnt3_LEV{config['main']['level']}/"

    OUTPUT_DIR = config['main']['result_dir'] + "predmap/"
    os.makedirs(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) is False else None

    # predmapを作成済みのWSIはスキップ
    skip_list = [predmap_fname.replace(".png", "") for predmap_fname in os.listdir(OUTPUT_DIR)]
    skip_list = list(set(skip_list))
    # ========================================================== #

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    project = (
        config['main']['model']
        + "_" + config['main']['optim']
        + "_batch" + str(config['main']['batch_size'])
        + "_shape" + str(config['main']['shape']))

    weight_dir = config['test']['weight_dir']
    weight_list = [weight_dir + name for name in config['test']['weight_names']]

    for cv_num in range(config['main']['cv']):

        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"{config['test']['target_data']}_{config['main']['facility']}_wsi.jb"
        )

        net = build_model(
            config['main']['model'],
            num_classes=len(config['main']['classes'])
        )

        weight_path = weight_list[cv_num]
        logging.info("Loading model {}".format(weight_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(
            torch.load(weight_path, map_location=device))

        net.eval()
        for wsi_name in wsis:
            logging.info(f"== {wsi_name} ==")

            # 既に作成済みのwsiはスキップ
            tmp_skip_list = [s for s in skip_list if s in wsi_name]
            if len(tmp_skip_list) > 0:
                print(f"skip: {wsi_name}")
                continue

            PMAP = makePredmap(
                wsi_name=wsi_name,
                classes=config['main']['classes'],
                level=config['main']['level'],
                wsi_dir=WSI_DIR,
                overlaid_mask_dir=MASK_DIR
            )

            patch_list = natsorted(glob.glob(PATCH_DIR + f"/{wsi_name}/*.png", recursive=False))

            test_data = WSI(
                patch_list,
                config['main']['classes'],
                tuple(config['main']['shape']),
                transform={'Resize': True, 'HFlip': False, 'VFlip': False},
                is_pred=True
            )

            loader = DataLoader(
                test_data, batch_size=config['main']['batch_size'],
                shuffle=False, num_workers=0, pin_memory=True)

            n_val = len(loader)  # the number of batch

            all_preds = []
            logging.info("predict class...")
            with tqdm(total=n_val, desc='prediction-map', unit='batch', leave=False) as pbar:
                for batch in loader:
                    imgs = batch['image']
                    imgs = imgs.to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        preds = net(imgs)
                    preds = nn.Softmax(dim=1)(preds).to('cpu').detach()
                    all_preds.extend(preds)

                    pbar.update()

            # 予測結果からセグメンテーション画像を生成
            logging.info("make segmented image from prediction results ...")
            PMAP.preds_to_image(
                preds=all_preds,
                output_dir=OUTPUT_DIR,
                output_name=wsi_name
            )

            # 背景&対象外領域をマスク
            logging.info("mask bg & other classes area...")
            PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
