import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eval import get_confusion_matrix, get_miss_preds


# for train mode
def eval_net(net, loader, criterion, device):
    net.eval()

    n_val = len(loader)  # the number of batch
    total_loss = 0
    init_flag = True

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_0 = batch['image_0']
            imgs_1 = batch['image_1']
            imgs_2 = batch['image_2']
            labels = batch['label']

            imgs_0 = imgs_0.to(device=device, dtype=torch.float32)
            imgs_1 = imgs_1.to(device=device, dtype=torch.float32)
            imgs_2 = imgs_2.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                preds = net(imgs_0, imgs_1, imgs_2)

            total_loss += criterion(preds, labels).item()

            # confusion matrix
            preds = nn.Softmax(dim=1)(preds)
            if init_flag:
                cm = get_confusion_matrix(preds, labels)
                init_flag = False
            else:
                cm += get_confusion_matrix(preds, labels)

            pbar.update()

    net.train()
    return total_loss / n_val, cm


# for test mode
def eval_net_test(net, loader, criterion, device, get_miss=False, save_dir=None):
    net.eval()

    n_val = len(loader)  # the number of batch
    total_loss = 0
    init_flag = True

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        batch_idx = 0
        for batch in loader:
            imgs_0 = batch['image_0']
            imgs_1 = batch['image_1']
            imgs_2 = batch['image_2']
            labels = batch['label']

            imgs_0 = imgs_0.to(device=device, dtype=torch.float32)
            imgs_1 = imgs_1.to(device=device, dtype=torch.float32)
            imgs_2 = imgs_2.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                preds = net(imgs_0, imgs_1, imgs_2)

            total_loss += criterion(preds, labels).item()

            # confusion matrix
            preds = nn.Softmax(dim=1)(preds)
            if init_flag:
                cm = get_confusion_matrix(preds, labels)
                init_flag = False
            else:
                cm += get_confusion_matrix(preds, labels)

            if get_miss:
                get_miss_preds(preds, labels, batch['name'], imgs_0, save_dir=save_dir + "miss_predict/",
                               ext=str(batch_idx).zfill(3))

            batch_idx += 1
            pbar.update()

    net.train()
    return total_loss / n_val, cm
