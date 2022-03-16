import os
import sys
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
import re
import pathlib


def get_files_dic(wsis: list, levels: int, classes: list, imgs_dir: str):
    files_dic = {}
    for i in range(len(levels)):
        files_dic[i] = get_files(wsis, level=levels[i], classes=classes, imgs_dir=imgs_dir)
    return files_dic


def get_files(wsis: list, level: int, classes: list, imgs_dir: str):
    def get_sub_classes(classes):
        # classesからsub-classを取得
        sub_cl_list = []
        for idx in range(len(classes)):
            cl = classes[idx]
            if isinstance(cl, list):
                for sub_cl in cl:
                    sub_cl_list.append(sub_cl)
            else:
                sub_cl_list.append(cl)
        return sub_cl_list

    re_pattern = re.compile('|'.join([f"/{i}/" for i in get_sub_classes(classes)]))

    files_list = []
    for wsi in wsis:
        files_list.extend(
            [
                p for p in glob.glob(imgs_dir + f"*/{wsi}_*/{level}/*.png", recursive=True)
                if bool(re_pattern.search(p))
            ]
        )
    return files_list


class WSI_multi(torch.utils.data.Dataset):
    def __init__(
            self,
            file_list_0: list,
            file_list_1: list,
            file_list_2: list,
            classes: list = [0, 1, 2, 3],
            shape: tuple = None,
            transform=None,
            is_pred: bool = False):
        """
        For multi-scale model
        """
        self.file_list_0 = file_list_0
        self.file_list_1 = file_list_1
        self.file_list_2 = file_list_2
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.is_pred = is_pred

    def __len__(self):
        return len(self.file_list_0)

    # pathからlabelを取得
    # TODO: /{level}/と/{cl}/が被る可能性があるため要修正
    def get_label(self, path):
        def check_path(cl, path):
            if f"/{cl}/" in path:
                return True
            else:
                return False

        # pathに/{cl}/と/{level}/の両方が含まれるため，/{level}/部を削除
        path = str(pathlib.Path(path).parents[1])

        for idx in range(len(self.classes)):
            cl = self.classes[idx]

            if isinstance(cl, list):
                for sub_cl in cl:
                    if check_path(sub_cl, path):
                        label = idx
            else:
                if check_path(cl, path):
                    label = idx
        assert label is not None, "label is not included in {path}"
        return np.array(label)

    def preprocess(self, img_pil):
        if self.transform is not None:
            if self.transform['Resize']:
                img_pil = transforms.Resize(
                    self.shape
                )(img_pil)
            if self.transform['HFlip']:
                img_pil = transforms.RandomHorizontalFlip(0.5)(img_pil)
            if self.transform['VFlip']:
                img_pil = transforms.RandomVerticalFlip(0.5)(img_pil)
        return np.asarray(img_pil)

    def transpose(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # For rgb or grayscale image
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def get_img(self, file_path: str):
        img_pil = Image.open(file_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img = self.preprocess(img_pil)
        img = self.transpose(img)
        return img

    def __getitem__(self, i):
        img_file_0 = self.file_list_0[i]
        img_0 = self.get_img(self.file_list_0[i])
        img_1 = self.get_img(self.file_list_1[i])
        img_2 = self.get_img(self.file_list_2[i])

        if self.is_pred:
            item = {
                'image_0': torch.from_numpy(img_0).type(torch.FloatTensor),
                'image_1': torch.from_numpy(img_1).type(torch.FloatTensor),
                'image_2': torch.from_numpy(img_2).type(torch.FloatTensor),
                'name': img_file_0
            }
        else:
            label = self.get_label(img_file_0)
            item = {
                'image_0': torch.from_numpy(img_0).type(torch.FloatTensor),
                'image_1': torch.from_numpy(img_1).type(torch.FloatTensor),
                'image_2': torch.from_numpy(img_2).type(torch.FloatTensor),
                'label': torch.from_numpy(label).type(torch.long),
                'name': img_file_0
            }

        return item


class WSIDataset_multi(object):
    def __init__(
        self,
        imgs_dir: str,
        train_wsis: list = None,
        valid_wsis: list = None,
        test_wsis: list = None,
        classes: list = [0, 1, 2],
        shape: tuple = (512, 512),
        transform: dict = None,
        levels: list = [0, 1, 2],
    ):
        """
        For multi-scale model
        """
        self.train_wsis = train_wsis
        self.valid_wsis = valid_wsis
        self.test_wsis = test_wsis

        self.imgs_dir = imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.levels = levels
        self.sub_classes = self.get_sub_classes()

        # self.wsi_list = []
        # for i in range(len(self.sub_classes)):
        #     sub_cl = self.sub_classes[i]
        #     self.wsi_list.extend([p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")])
        # self.wsi_list = list(set(self.wsi_list))
        # # os.listdirによる実行時における要素の順不同対策のため
        # self.wsi_list = natsorted(self.wsi_list)

        self.train_files_dic = self.get_files_dic(self.train_wsis, self.levels)
        self.valid_files_dic = self.get_files_dic(self.valid_wsis, self.levels)
        self.test_files_dic = self.get_files_dic(self.test_wsis, self.levels)

        print(f"[wsi]  train: {len(self.train_wsis)}, valid: {len(self.valid_wsis)}, test: {len(self.test_wsis)}")

        self.data_len = len(self.train_files_dic[0]) + len(self.valid_files_dic[0]) + len(self.test_files_dic[0])
        print(f"[data] train: {len(self.train_files_dic[0])}, valid: {len(self.valid_files_dic[0])}, test: {len(self.test_files_dic[0])}")

        for i in range(len(self.test_files_dic)):
            self.test_files_dic[i] = natsorted(self.test_files_dic[i])

        self.train_data = WSI_multi(self.train_files_dic[0], self.train_files_dic[1], self.train_files_dic[2], self.classes, self.shape, self.transform)

        test_transform = self.transform.copy()
        test_transform['HFlip'] = False
        test_transform['VFlip'] = False
        self.valid_data = WSI_multi(self.valid_files_dic[0], self.valid_files_dic[1], self.valid_files_dic[2], self.classes, self.shape, test_transform)
        self.test_data = WSI_multi(self.test_files_dic[0], self.test_files_dic[1], self.test_files_dic[2], self.classes, self.shape, test_transform)

    def __len__(self):
        return len(self.data_len)

    def get_sub_classes(self):
        # classesからsub-classを取得
        sub_cl_list = []
        for idx in range(len(self.classes)):
            cl = self.classes[idx]
            if isinstance(cl, list):
                for sub_cl in cl:
                    sub_cl_list.append(sub_cl)
            else:
                sub_cl_list.append(cl)
        return sub_cl_list

    def get_files_dic(self, wsis: list, levels: list):
        files_dic = {}
        for i in range(len(levels)):
            files_dic[i] = natsorted(self.get_files(wsis, level=levels[i]))
        return files_dic

    def get_files(self, wsis: list, level: int):
        re_pattern = re.compile('|'.join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p for p in glob.glob(self.imgs_dir + f"*/{wsi}_*/{level}/*.png", recursive=True)
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list

    def get(self):
        return self.train_data, self.valid_data, self.test_data
