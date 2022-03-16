import logging
import os
import sys
import yaml
import joblib
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from natsort import natsorted
import pathlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src_multi.dataset import WSI_multi
from src_multi.model import MultiscaleNet
from src.predict_faster import makePredmap
from src.util import fix_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    fix_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='../config/config_src_LEV012.yaml')
    parser.add_argument('--main_dir', default='/mnt/ssdsam/chemotherapy_strage/')
    parser.add_argument('--batch_size', default=256)

    args = parser.parse_args()
    config_path = args.config_path
    MAIN_DIR = args.main_dir

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    # ========================================================== #
    WSI_DIR = MAIN_DIR + "mnt1/origin/"
    MASK_DIR = MAIN_DIR + f"mnt1/mask_cancergrade/overlaid_{config['main']['classes']}/"
    # PATCH_DIR = MAIN_DIR + "mnt3_LEV012/"

    PATCH_DIR = MAIN_DIR.replace("ssdsam/", "ssdwdc/") + "mnt3_LEV012/"

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

    weight_dir = config['test']['weight_dir']
    weight_list = [weight_dir + name for name in config['test']['weight_names']]

    for cv_num in range(config['main']['cv']):

        if cv_num != 2:
            continue

        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"{config['test']['target_data']}_wsi.jb"
        )

        net = MultiscaleNet(
            base_fe=config['main']['model'],
            num_class=len(config['main']['classes']))

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
                level=config['main']['levels'][0],
                wsi_dir=WSI_DIR,
                overlaid_mask_dir=MASK_DIR
            )

            # patch_list_0 = natsorted(
            #     glob.glob(PATCH_DIR + f"/{wsi_name}/{config['main']['levels'][0]}/*.png", recursive=False))
            # patch_list_1 = natsorted(
            #     glob.glob(PATCH_DIR + f"/{wsi_name}/{config['main']['levels'][1]}/*.png", recursive=False))
            # patch_list_2 = natsorted(
            #     glob.glob(PATCH_DIR + f"/{wsi_name}/{config['main']['levels'][2]}/*.png", recursive=False))

            patch_parent = pathlib.Path(PATCH_DIR)
            patch_list_0 = natsorted(
                list(patch_parent.glob(f"{wsi_name}/{config['main']['levels'][0]}/*.png")))
            patch_list_1 = natsorted(
                list(patch_parent.glob(f"{wsi_name}/{config['main']['levels'][1]}/*.png")))
            patch_list_2 = natsorted(
                list(patch_parent.glob(f"{wsi_name}/{config['main']['levels'][2]}/*.png")))
            patch_list_0 = [str(path) for path in patch_list_0]
            patch_list_1 = [str(path) for path in patch_list_1]
            patch_list_2 = [str(path) for path in patch_list_2]

            test_data = WSI_multi(
                file_list_0=patch_list_0,
                file_list_1=patch_list_1,
                file_list_2=patch_list_2,
                classes=config['main']['classes'],
                shape=tuple(config['main']['shape']),
                transform={'Resize': True, 'HFlip': False, 'VFlip': False},
                is_pred=True
            )

            loader = DataLoader(
                test_data, batch_size=args.batch_size,
                shuffle=False, num_workers=0, pin_memory=True)

            n_val = len(loader)  # the number of batch

            all_preds = []
            logging.info("predict class...")
            with tqdm(total=n_val, desc='prediction-map', unit='batch', leave=False) as pbar:
                for batch in loader:
                    imgs_0 = batch['image_0']
                    imgs_1 = batch['image_1']
                    imgs_2 = batch['image_2']
                    imgs_0 = imgs_0.to(device=device, dtype=torch.float32)
                    imgs_1 = imgs_1.to(device=device, dtype=torch.float32)
                    imgs_2 = imgs_2.to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        preds = net(imgs_0, imgs_1, imgs_2)
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
