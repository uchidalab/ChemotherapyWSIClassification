import os
import sys
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
import re


def get_files(wsis, classes, imgs_dir):
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
                p for p in glob.glob(imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                if bool(re_pattern.search(p))
            ]
        )
    return files_list


class WSI(torch.utils.data.Dataset):
    def __init__(self, file_list, classes=[0, 1, 2, 3], shape=None, transform=None, is_pred=False):
        self.file_list = file_list
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.is_pred = is_pred

    def __len__(self):
        return len(self.file_list)

    # pathからlabelを取得
    def get_label(self, path):
        def check_path(cl, path):
            if f"/{cl}/" in path:
                return True
            else:
                return False

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

    def __getitem__(self, i):
        img_file = self.file_list[i]
        img_pil = Image.open(img_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        img = self.preprocess(img_pil)
        img = self.transpose(img)

        if self.is_pred:
            item = {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'name': img_file
            }
        else:
            label = self.get_label(img_file)
            item = {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'label': torch.from_numpy(label).type(torch.long),
                'name': img_file
            }

        return item


class WSIDataset(object):
    def __init__(
        self,
        imgs_dir: str,
        train_wsis: list=None,
        valid_wsis: list=None,
        test_wsis: list=None,
        train_files: list=None,
        valid_files: list=None,
        test_files: list=None,
        classes: list=[0, 1, 2, 3],
        shape: tuple=(512, 512),
        transform: dict=None,
    ):
        self.train_wsis = train_wsis
        self.valid_wsis = valid_wsis
        self.test_wsis = test_wsis

        self.train_files = train_files
        self.valid_files = valid_files
        self.test_files = test_files

        self.imgs_dir = imgs_dir
        self.classes = classes
        self.shape = shape
        self.transform = transform
        self.sub_classes = self.get_sub_classes()

        # self.wsi_list = []
        # for i in range(len(self.sub_classes)):
        #     sub_cl = self.sub_classes[i]
        #     self.wsi_list.extend([p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")])
        # self.wsi_list = list(set(self.wsi_list))
        # # os.listdirによる実行時における要素の順不同対策のため
        # self.wsi_list = natsorted(self.wsi_list)

        if ((self.train_wsis is not None)
            and (self.valid_wsis is not None)
            and (self.valid_wsis is not None)
        ):
            self.train_files = self.get_files(self.train_wsis)
            self.valid_files = self.get_files(self.valid_wsis)
            self.test_files = self.get_files(self.test_wsis)
            print(f"[wsi]  train: {len(self.train_wsis)}, valid: {len(self.valid_wsis)}, test: {len(self.test_wsis)}")
        elif ((self.train_files is not None)
            and (self.valid_files is not None)
            and (self.test_files is not None)
        ):
            pass
        else:
            sys.exit("wsis lists or files lists are not given")

        self.data_len = len(self.train_files) + len(self.valid_files) + len(self.test_files)
        print(f"[data] train: {len(self.train_files)}, valid: {len(self.valid_files)}, test: {len(self.test_files)}")

        self.test_files = natsorted(self.test_files)

        self.train_data = WSI(self.train_files, self.classes, self.shape, self.transform)

        test_transform = self.transform.copy()
        test_transform['HFlip'] = False
        test_transform['VFlip'] = False
        self.valid_data = WSI(self.valid_files, self.classes, self.shape, test_transform)
        self.test_data = WSI(self.test_files, self.classes, self.shape, test_transform)

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

    def get_files(self, wsis):
        re_pattern = re.compile('|'.join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p for p in glob.glob(self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list

    def get(self):
        return self.train_data, self.valid_data, self.test_data
