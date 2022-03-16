import logging
import os
import sys
import yaml
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import WSI, get_files
from src.eval import eval_net_test, plot_confusion_matrix, eval_metrics
from src.util import fix_seed
from src.model import build_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    fix_seed(0)
    # config_path = './config/config_src_LEV1.yaml'
    config_path = '../config/config_src_LEV1.yaml'
    # config_path = '../config/config_src_LEV2.yaml'

    font_size = 35
    cl_labels = ["Non-\nNeop.", "Tumor\nbed", "Residual\ntumor"]

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_dir = config['test']['weight_dir']
    weight_list = [weight_dir + name for name in config['test']['weight_names']]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}{config['test']['target_data']}.txt",
        format='%(levelname)s: %(message)s'
    )

    for cv_num in range(config['main']['cv']):
        logging.info(f"== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"cv{cv_num}_"

        project = (
            project_prefix
            + config['main']['model']
            + "_" + config['main']['optim']
            + "_batch" + str(config['main']['batch_size'])
            + "_shape" + str(config['main']['shape'])
            + "_lev" + str(config['main']['level']))
        logging.info(f"{project}\n")

        criterion = nn.CrossEntropyLoss()

        net = build_model(
            config['main']['model'],
            num_classes=len(config['main']['classes'])
        )

        logging.info("Loading model {}".format(weight_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(
            torch.load(weight_path, map_location=device))

        # files = joblib.load(
        #     config['dataset']['jb_dir']
        #     + f"cv{cv_num}_"
        #     + f"{config['test']['target_data']}.jb"
        # )

        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"{config['test']['target_data']}_wsi.jb"
        )
        files = get_files(
            wsis=wsis,
            classes=config['main']['classes'],
            imgs_dir=config['dataset']['imgs_dir'])

        dataset = WSI(
            files,
            config['main']['classes'],
            tuple(config['main']['shape']),
            transform={'Resize': True, 'HFlip': False, 'VFlip': False}
        )

        loader = DataLoader(
            dataset, batch_size=config['main']['batch_size'],
            shuffle=False, num_workers=2, pin_memory=True)
        val_loss, cm = eval_net_test(
            net, loader, criterion, device,
            get_miss=config['test']['get_miss'],
            save_dir=config['test']['output_dir'])

        if cv_num == 0:
            cm_all = cm
        else:
            cm_all += cm

        logging.info(
            f"\n cm ({config['test']['target_data']}):\n{np.array2string(cm, separator=',')}\n"
        )
        val_metrics = eval_metrics(cm)
        logging.info('===== eval metrics =====')
        logging.info(f"\n Accuracy ({config['test']['target_data']}):  {val_metrics['accuracy']}")
        logging.info(f"\n Precision ({config['test']['target_data']}): {val_metrics['precision']}")
        logging.info(f"\n Recall ({config['test']['target_data']}):    {val_metrics['recall']}")
        logging.info(f"\n F1 ({config['test']['target_data']}):        {val_metrics['f1']}")
        logging.info(f"\n mIoU ({config['test']['target_data']}):      {val_metrics['mIoU']}")

        # Not-Normalized
        cm_plt = plot_confusion_matrix(
            cm, cl_labels, normalize=False, font_size=25)
        cm_plt.savefig(
            config['test']['output_dir']
            + project
            + f"_{config['test']['target_data']}_nn-confmatrix.png"
        )
        plt.clf()
        plt.close()

        # Normalized
        cm_plt = plot_confusion_matrix(
            cm, cl_labels, normalize=True, font_size=font_size)
        cm_plt.savefig(
            config['test']['output_dir']
            + project
            + f"_{config['test']['target_data']}_confmatrix.png"
        )
        plt.clf()
        plt.close()

    # ===== cv_all ===== #
    logging.info(f"\n\n== ALL ==")
    project_prefix = "all_"

    project = (
        project_prefix
        + config['main']['model']
        + "_" + config['main']['optim']
        + "_batch" + str(config['main']['batch_size'])
        + "_shape" + str(config['main']['shape'])
        + "_lev" + str(config['main']['level']))
    logging.info(f"{project}\n")

    logging.info(
        f"\n cm ({config['test']['target_data']}):\n{np.array2string(cm_all, separator=',')}\n"
    )
    val_metrics = eval_metrics(cm_all)
    logging.info('===== eval metrics =====')
    logging.info(f"\n Accuracy ({config['test']['target_data']}):  {val_metrics['accuracy']}")
    logging.info(f"\n Precision ({config['test']['target_data']}): {val_metrics['precision']}")
    logging.info(f"\n Recall ({config['test']['target_data']}):    {val_metrics['recall']}")
    logging.info(f"\n F1 ({config['test']['target_data']}):        {val_metrics['f1']}")
    logging.info(f"\n mIoU ({config['test']['target_data']}):      {val_metrics['mIoU']}")

    # Not-Normalized
    cm_plt = plot_confusion_matrix(
        cm_all, cl_labels, normalize=False, font_size=25)
    cm_plt.savefig(
        config['test']['output_dir']
        + project
        + f"_{config['test']['target_data']}_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(
        cm_all, cl_labels, normalize=True, font_size=font_size)
    cm_plt.savefig(
        config['test']['output_dir']
        + project
        + f"_{config['test']['target_data']}_confmatrix.png"
    )
    plt.clf()
    plt.close()
