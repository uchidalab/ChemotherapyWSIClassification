import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.util import make_grid_images
from src.metrics import evalMet


# for train mode
def eval_net(net, loader, criterion, device):
    net.eval()

    n_val = len(loader)  # the number of batch
    total_loss = 0
    init_flag = True

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, labels = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                preds = net(imgs)

            total_loss += criterion(preds, labels).item()

            # confusion matrix
            if net.fc.out_features > 1:
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
            imgs, labels = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                preds = net(imgs)

            total_loss += criterion(preds, labels).item()

            # confusion matrix
            if net.fc.out_features > 1:
                preds = nn.Softmax(dim=1)(preds)
            if init_flag:
                cm = get_confusion_matrix(preds, labels)
                init_flag = False
            else:
                cm += get_confusion_matrix(preds, labels)

            if get_miss:
                get_miss_preds(preds, labels, batch['name'], imgs, save_dir=save_dir + "miss_predict/",
                               ext=str(batch_idx).zfill(3))

            batch_idx += 1
            pbar.update()

    net.train()
    return total_loss / n_val, cm


def eval_metrics(cm):
    Met = evalMet()
    met_val = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mIoU': 0}
    met_val['accuracy'] = Met.Accuracy(cm)
    met_val['precision'] = Met.Precision(cm)
    met_val['recall'] = Met.Recall(cm)
    met_val['f1'] = Met.F1(cm)
    met_val['mIoU'] = Met.mIoU(cm)
    return met_val


# preds: [N, C], targs: [N] (N means batch-size)
def get_confusion_matrix(preds, targs):
    num_classes = preds.shape[1]
    preds = preds.argmax(dim=1)

    assert preds.shape == targs.shape, f'predict & target shape do not match\n \
        preds: {preds.shape}, targs: {targs.shape}'

    preds = preds.cpu()
    targs = targs.cpu()
    labels = [i for i in range(num_classes)]
    cm = confusion_matrix(targs, preds, labels=labels)
    return cm


def plot_confusion_matrix(cm, class_names, normalize=True, font_size=20):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    figure = plt.figure(figsize=(8, 8))
    plt.rcParams["font.size"] = font_size

    # Normalize the confusion matrix.
    if normalize:
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # plt.title("Confusion matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure


def convert_plt2nd(figure):
    figure.canvas.draw()
    img = np.array(figure.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def get_miss_preds(preds, labels, names, imgs=None, save_dir=None, ext=""):
    """
    Args:
       preds: pred results after softmax & argmax
       labels: label list (batch['label'])
       names: name list (batch['name'])
    """

    preds = preds.argmax(dim=1)

    if preds.device.type != "cpu":
        preds = preds.cpu()
    if labels.device.type != "cpu":
        labels = labels.cpu()

    miss_list = []
    for idx in range(len(labels)):
        pred = preds[idx].item()
        label = labels[idx].item()
        name = names[idx]

        if pred != label:
            print(f"[miss] {name} pred: {pred}, label: {label}")
            miss_list.append(idx)

    if (save_dir is not None) and (len(miss_list) > 0):
        assert imgs is not None, "imgs should be set if save_dir is not None"
        make_grid_images(miss_list, imgs, preds, labels, names, save_dir, ext)

    return miss_list
