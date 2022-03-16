import os
import sys
import numpy as np
import cv2
import random
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import WSI
from annotation.util import num_to_color


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    # 最新版はtorch.cuda.nanual_seed_all()は不要
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def select_optim(optim_name, net_params):
    if optim_name == "Adam":
        optim = torch.optim.Adam(net_params)
    elif optim_name == "Adagrad":
        optim = torch.optim.Adagrad(net_params)
    elif optim_name == "AdamW":
        optim = torch.optim.AdamW(net_params)
    else:
        raise Exception("Unexpected optim_name: {}".format(optim_name))
    return optim


# ch=class_num -> rgb
# img shape: (ch, h, w)
def contract_img_dim(img):
    mask = np.full((img.shape[1], img.shape[2]), 255)
    # 背景クラスを含む
    class_num_all = img.shape[0]
    # 背景クラス除く
    class_num = class_num_all - 1

    for i in range(class_num_all):
        mask[img[i, :, :] == 1] = i

    canvas = np.full((img.shape[1], img.shape[2], 3), 255)
    for i in range(class_num):
        canvas[mask == i] = num_to_color(i)
    return canvas


def put_label2img(img, pred_label, true_label, name=None, is_transpose=False, is_mul=True):
    org_coord = (10, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (10, 10, 10)
    thickness = 1

    if name is not None:
        texts = f'{name}_T:{str(true_label)}_P:{str(pred_label)}'
    else:
        texts = f'T:{str(true_label)}_P:{str(pred_label)}'
    if is_transpose:
        img = img.transpose(1, 2, 0)
    if is_mul:
        img = img * 255
    img = np.uint8(img)
    tmp_array = cv2.putText(
        img, texts, org_coord, font, font_scale, color, thickness, cv2.LINE_AA)
    tmp_array = cv2.UMat.get(tmp_array)
    return tmp_array


def make_grid_images(idx_list, imgs, preds, labels, names, save_dir, ext=""):
    imgs_list = []
    for idx in idx_list:
        pred_label = preds[idx].item()
        true_label = labels[idx].item()
        img = imgs[idx].cpu().numpy()
        img = put_label2img(
            img, pred_label, true_label, name=None, is_transpose=True, is_mul=True)
        imgs_list.append(img)
    imgs = np.array(imgs_list)
    # save_imageで255掛けるため[0,1]スケールにしておく
    imgs = (imgs / 255.0).astype(np.float32)
    # NHWC -> NCHW (for torch tensor)
    imgs_tensor = torch.from_numpy(np.transpose(imgs, [0, 3, 1, 2]))
    nrow = len(imgs_list) if 5 > len(imgs_list) else 5
    torchvision.utils.save_image(imgs_tensor, f"{save_dir}miss_predict_{ext}.png", nrow=nrow, padding=10)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif isinstance(dataset, WSI):
            return dataset[idx]['label'].item()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
