import os
import sys
import yaml
import joblib
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import collections

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.train import update_best_model, early_stop
from src.eval import plot_confusion_matrix, convert_plt2nd, eval_metrics
from src.util import fix_seed, put_label2img, select_optim
from src_multi.dataset import WSIDataset_multi
from src_multi.model import MultiscaleNet
from src_multi.eval import eval_net
from src_multi.util import ImbalancedDatasetSampler


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_net(net,
              train_data,
              valid_data,
              device,
              epochs=5,
              batch_size=4,
              optim_name="Adam",
              save_cp=True,
              classes=[0, 1, 2],
              checkpoint_dir="checkpoints/",
              writer=None,
              patience=5,
              stop_cond='mIoU',
              mode='max',
              cv_num=0):

    n_train = len(train_data)

    train_loader = DataLoader(
        train_data,
        sampler=ImbalancedDatasetSampler(train_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_data,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    optimizer = select_optim(optim_name, net.parameters())

    criterion = nn.CrossEntropyLoss()

    if mode == 'min':
        best_model_info = {'epoch': 0, 'val': float('inf')}
    elif mode == 'max':
        best_model_info = {'epoch': 0, 'val': float('-inf')}

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        counter = 1
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            count_label = []  # to check imblanaced sampler
            for batch in train_loader:
                imgs_0 = batch['image_0']
                imgs_1 = batch['image_1']
                imgs_2 = batch['image_2']
                labels = batch['label']
                # names = batch['name']

                imgs_0 = imgs_0.to(device=device, dtype=torch.float32)
                imgs_1 = imgs_1.to(device=device, dtype=torch.float32)
                imgs_2 = imgs_2.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                preds = net(imgs_0, imgs_1, imgs_2)

                loss = criterion(preds, labels)

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs_0.shape[0])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                count_label.extend(labels.tolist())  # to check sampler
        print(f"sample balance: {collections.Counter(count_label)}")  # to check sampler

        # calculate validation loss and confusion matrix
        val_loss, cm = eval_net(net, val_loader, criterion, device)

        # calculate validation metircs
        val_metrics = eval_metrics(cm)
        cond_val = val_metrics[stop_cond]

        best_model_info = update_best_model(cond_val, epoch, best_model_info, mode=mode)
        logging.info('\n Loss (train, epoch): {}'.format(epoch_loss))
        logging.info('\n Loss (valid, batch): {}'.format(val_loss))
        logging.info('\n mIoU (valid, epoch): {}'.format(val_metrics['mIoU']))

        if writer is not None:
            # upload loss (train) and learning_rate to tensorboard
            writer.add_scalar('Loss/train', epoch_loss, epoch)

            # upload confusion_matrix (validation) to tensorboard
            cm_plt = plot_confusion_matrix(cm, classes, normalize=True)
            cm_nd = convert_plt2nd(cm_plt)
            writer.add_image(
                'confusion_matrix/valid',
                cm_nd,
                global_step=epoch,
                dataformats='HWC'
            )
            plt.clf()
            plt.close()

            # upload not-normed confusion_matrix (validation) to tensorboard
            cm_plt = plot_confusion_matrix(cm, classes, normalize=False)
            cm_nd = convert_plt2nd(cm_plt)
            writer.add_image(
                'confusion_matrix_nn/valid',
                cm_nd,
                global_step=epoch,
                dataformats='HWC'
            )
            plt.clf()
            plt.close()

            # upload loss & score (validation) to tensorboard
            writer.add_scalar('Loss/valid', val_loss, epoch)
            writer.add_scalar('mIoU/valid', val_metrics['mIoU'], epoch)
            writer.add_scalar('Accuracy/valid', val_metrics['accuracy'], epoch)
            writer.add_scalar('Precision/valid', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/valid', val_metrics['recall'], epoch)
            writer.add_scalar('F1/valid', val_metrics['f1'], epoch)

        counter += 1

        if save_cp:
            try:
                os.mkdir(checkpoint_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            if best_model_info['epoch'] == epoch:
                torch.save(
                    net.state_dict(),
                    checkpoint_dir + f'cv{cv_num}_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

        if early_stop(cond_val, epoch, best_model_info, patience=patience, mode=mode):
            break

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    fix_seed(0)
    # config_path = './config/config_src_LEV012.yaml'
    config_path = '../config/config_src_LEV012.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    input_shape = tuple(config['main']['shape'])
    transform = {'Resize': True, 'HFlip': True, 'VFlip': True}

    # # 3つの倍率の事前学習済み重み
    # weight_list_0 = [config['main']['weight_dir_0'] + name for name in config['main']['weight_names_0']]
    # weight_list_1 = [config['main']['weight_dir_1'] + name for name in config['main']['weight_names_1']]
    # weight_list_2 = [config['main']['weight_dir_2'] + name for name in config['main']['weight_names_2']]

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    for cv_num in range(config['main']['cv']):
        logging.info(f'== CV{cv_num} ==')
        writer = SummaryWriter(
            log_dir=(
                f"{config['main']['result_dir']}logs/{config['main']['model']}_{config['main']['optim']}_batch{config['main']['batch_size']}_shape{config['main']['shape']}_cl{config['main']['classes']}_lev{config['main']['levels']}_cv{cv_num}"
            )
        )

        net = MultiscaleNet(
            base_fe=config['main']['model'],
            num_class=len(config['main']['classes']),
            pretrained=True,
            weight_path_0=None,
            weight_path_1=None,
            weight_path_2=None)
        net.to(device=device)

        train_wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"train_wsi.jb"
        )
        valid_wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"valid_wsi.jb"
        )
        test_wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"test_wsi.jb"
        )

        dataset = WSIDataset_multi(
            imgs_dir=config['dataset']['imgs_dir'],
            train_wsis=train_wsis,
            valid_wsis=valid_wsis,
            test_wsis=test_wsis,
            classes=config['main']['classes'],
            shape=input_shape,
            transform=transform,
            levels=config['main']['levels']
        )
        train_data, valid_data, test_data = dataset.get()

        logging.info(f'''Starting training:
            Classes:           {config['main']['classes']}
            Levels:            {config['main']['levels']}
            Epochs:            {config['main']['epochs']}
            Batch size:        {config['main']['batch_size']}
            Model:             {config['main']['model']}
            Optim:             {config['main']['optim']}
            Transform:         {json.dumps(transform)}
            Training size:     {len(train_data)}
            Validation size:   {len(valid_data)}
            Patience:          {config['main']['patience']}
            StopCond:          {config['main']['stop_cond']}
            Device:            {device.type}
            Images Shape:      {input_shape}
        ''')

        try:
            train_net(net=net,
                      train_data=train_data,
                      valid_data=valid_data,
                      epochs=config['main']['epochs'],
                      batch_size=config['main']['batch_size'],
                      device=device,
                      classes=config['main']['classes'],
                      checkpoint_dir=f"{config['main']['result_dir']}checkpoints/{config['main']['classes']}/",
                      writer=writer,
                      patience=config['main']['patience'],
                      stop_cond=config['main']['stop_cond'],
                      mode='max',
                      cv_num=cv_num)
        except KeyboardInterrupt:
            torch.save(
                net.state_dict(), config['main']['result_dir'] + f'cv{cv_num}_INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
