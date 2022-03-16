import os
import joblib
import logging
import glob
from natsort import natsorted
import re
import random
import copy
import math
from sklearn.model_selection import train_test_split

'''
splitWSIDataset:
    imgs_dirにある予測対象のクラスのWSIのみから，
    Cross Validation用にデータセットを分割する

ディレクトリの構造 (例): /{imgs_dir}/{sub_cl}/{wsi_name}/0_0000000.png
'''


class splitWSIDataset(object):
    def __init__(self, imgs_dir, classes=[0, 1, 2, 3], val_ratio=0.2, random_seed=0):
        self.imgs_dir = imgs_dir
        self.classes = classes
        self.val_ratio = val_ratio
        self.sub_classes = self.get_sub_classes()
        self.random_seed = random_seed
        self.sets_num = 5

        random.seed(self.random_seed)

        # WSIごとにtrain, valid, test分割
        self.wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            self.wsi_list.extend([p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")])
        self.wsi_list = list(set(self.wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        self.wsi_list = natsorted(self.wsi_list)

        # WSIのリストを5-setsに分割
        random.shuffle(self.wsi_list)
        self.sets_list = self.split_sets_list(self.wsi_list)

    def __len__(self):
        return len(self.wsi_list)

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

    def split_sets_list(self, wsi_list, sets_num=5):
        wsi_num = len(wsi_list)
        q, mod = divmod(wsi_num, sets_num)
        logging.info(f"wsi_num: {wsi_num}, q: {q}, mod: {mod}")

        idx_list = []
        wsi_sets = []
        idx = 0

        for cv in range(sets_num):
            if cv < mod:
                end_idx = idx + q
            else:
                end_idx = (idx + q) - 1
            idx_list.append([idx, end_idx])

            wsi_sets.append(wsi_list[idx:end_idx + 1])
            idx = end_idx + 1

        print(f"idx_list: {idx_list}")

        return wsi_sets

    def get_sets_list(self):
        return self.sets_list

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

    def get_cv_wsis(self, sets_list, cv_num):
        test_wsis = sets_list[cv_num]
        trvl_wsis = []
        for i in range(self.sets_num):
            if i == cv_num:
                continue
            else:
                trvl_wsis += sets_list[i]

        random.shuffle(trvl_wsis)
        train_wsis, valid_wsis = train_test_split(
            trvl_wsis, test_size=self.val_ratio, random_state=self.random_seed)
        return natsorted(train_wsis), natsorted(valid_wsis), natsorted(test_wsis)


def save_dataset(imgs_dir: str, output_dir: str, cv: int=5):
    dataset = splitWSIDataset(imgs_dir, classes=[0, 1, 2], val_ratio=0.2, random_seed=0)
    sets_list = dataset.get_sets_list()

    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")
        train_wsis, valid_wsis, test_wsis = dataset.get_cv_wsis(sets_list, cv_num=cv_num)

        train_files = dataset.get_files(train_wsis)
        valid_files = dataset.get_files(valid_wsis)
        test_files = dataset.get_files(test_wsis)

        logging.info(f"[wsi]  train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}")
        logging.info(f"[data] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(valid_wsis, output_dir + f"cv{cv_num}_valid_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        # 各データのリスト(path)を保存
        joblib.dump(train_files, output_dir + f"cv{cv_num}_train.jb", compress=3)
        joblib.dump(valid_files, output_dir + f"cv{cv_num}_valid.jb", compress=3)
        joblib.dump(test_files, output_dir + f"cv{cv_num}_test.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n")
            f.write(
                "== [patch] ==\n"
                + f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
                + "\n==============\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== valid (wsi) ==\n")
            for i in range(len(valid_wsis)):
                f.write(f"{valid_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


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

def get_files(wsis: list, imgs_dir: str, classes: list):
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


# chemotherapy dataset用 (train用WSIのパッチをランダムにvalidationに割当)
def save_dataset2(
    imgs_dir: str,
    output_dir: str,
    wsi_fold: list,
    classes: list=[0, 1, 2],
    cv: int=3,
    val_ratio: float=0.2,
    random_seed: int=0,
):
    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")
        wsi_fold_tmp = copy.deepcopy(wsi_fold)
        test_wsis = wsi_fold_tmp.pop(cv_num)
        train_wsis = [wsi_name for fold in wsi_fold_tmp for wsi_name in fold]
        logging.info(f"[wsi]  train: {len(train_wsis)}, test: {len(test_wsis)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        trvl_files = get_files(train_wsis, classes)
        test_files = get_files(test_wsis, classes)
        # trainに割り当てたWSIのパッチをランダムにtrain/validに分割
        train_files, valid_files = train_test_split(
            trvl_files, test_size=val_ratio, random_state=random_seed)
        logging.info(f"[patch]  train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

        # パッチ割当のリストを保存
        joblib.dump(train_files, output_dir + f"cv{cv_num}_train.jb", compress=3)
        joblib.dump(valid_files, output_dir + f"cv{cv_num}_valid.jb", compress=3)
        joblib.dump(test_files, output_dir + f"cv{cv_num}_test.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n")
            f.write(
                "== [patch] ==\n"
                + f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
                + "\n==============\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


# chemotherapy dataset用 (パッチ切り取り前に，WSIをtrain/validationに割り当て)
def save_dataset3(
    imgs_dir: str,
    output_dir: str,
    wsi_fold: list,
    classes: list=[0, 1, 2],
    cv: int=3,
    random_seed: int=0,
):
    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")
        wsi_fold_tmp = copy.deepcopy(wsi_fold)
        test_wsis = wsi_fold_tmp.pop(cv_num)
        valid_wsis = [fold.pop(0) for fold in wsi_fold_tmp]
        train_wsis = [wsi_name for fold in wsi_fold_tmp for wsi_name in fold]
        logging.info(f"[wsi]  train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(valid_wsis, output_dir + f"cv{cv_num}_valid_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        train_files = get_files(train_wsis, imgs_dir, classes)
        valid_files = get_files(valid_wsis, imgs_dir, classes)
        test_files = get_files(test_wsis, imgs_dir, classes)

        logging.info(f"[patch]  train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

        # パッチ割当のリストを保存
        joblib.dump(train_files, output_dir + f"cv{cv_num}_train.jb", compress=3)
        joblib.dump(valid_files, output_dir + f"cv{cv_num}_valid.jb", compress=3)
        joblib.dump(test_files, output_dir + f"cv{cv_num}_test.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n\n")
            f.write(
                "== [patch] ==\n"
                + f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
                + "\n==============\n\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== valid (wsi) ==\n")
            for i in range(len(valid_wsis)):
                f.write(f"{valid_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


# chemotherapy dataset用 (症例関係なくランダムにWSIをtrain/validation/testに割り当て)
def save_dataset4(
    imgs_dir: str,
    output_dir: str,
    wsi_fold: list,
    classes: list=[0, 1, 2],
    cv: int=3,
    val_ratio: float=0.2,
    random_seed: int=0,
):
    import numpy as np
    random.seed(random_seed)
    fold_idxs = [[], []]
    fold_idxs[0] = [i for i in range(len(wsi_fold[0]))]
    fold_idxs[1] = [i for i in range(len(wsi_fold[1]))]

    total_num = 0
    new_wsi_fold = [[], []]
    for j, fold in enumerate(wsi_fold):
        total_num += len(fold)
        random.shuffle(fold)

        fold_idxs[j] = [array.tolist() for array in np.array_split(fold_idxs[j], cv)]
        for idxs in fold_idxs[j]:
            new_wsi_fold[j].append([fold[idx] for idx in idxs])

    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")
        wsi_fold_tmp = copy.deepcopy(new_wsi_fold)

        test_wsis = wsi_fold_tmp[0].pop(cv_num)
        test_wsis += wsi_fold_tmp[1].pop(cv_num)

        wsi_fold_tmp0 = [wsi_name for fold in wsi_fold_tmp[0] for wsi_name in fold]
        wsi_fold_tmp1 = [wsi_name for fold in wsi_fold_tmp[1] for wsi_name in fold]
        random.shuffle(wsi_fold_tmp0)
        random.shuffle(wsi_fold_tmp1)

        train_wsis0, valid_wsis0 = train_test_split(
            wsi_fold_tmp0, test_size=val_ratio, random_state=random_seed)
        train_wsis1, valid_wsis1 = train_test_split(
            wsi_fold_tmp1, test_size=val_ratio, random_state=random_seed)

        train_wsis = train_wsis0 + train_wsis1
        valid_wsis = valid_wsis0 + valid_wsis1

        logging.info(f"[wsi]  train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(valid_wsis, output_dir + f"cv{cv_num}_valid_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        train_files = get_files(train_wsis, imgs_dir, classes)
        valid_files = get_files(valid_wsis, imgs_dir, classes)
        test_files = get_files(test_wsis, imgs_dir, classes)

        logging.info(f"[patch]  train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

        # パッチ割当のリストを保存
        joblib.dump(train_files, output_dir + f"cv{cv_num}_train.jb", compress=3)
        joblib.dump(valid_files, output_dir + f"cv{cv_num}_valid.jb", compress=3)
        joblib.dump(test_files, output_dir + f"cv{cv_num}_test.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n\n")
            f.write(
                "== [patch] ==\n"
                + f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
                + "\n==============\n\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== valid (wsi) ==\n")
            for i in range(len(valid_wsis)):
                f.write(f"{valid_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


def save_dataset_chemo_wsi(
    overlaid_dir: str = "/mnt/ssdsam/chemotherapy_strage/mnt1/mask_cancergrade/overlaid_[0, 1, 2]/",
    output_dir: str = "/mnt/ssdsam/chemotherapy_strage/dataset/202112_chemotherapy/",
    cv: int = 3,
    val_ratio: float=0.2,
    random_seed: int = 0,
):
    files = os.listdir(overlaid_dir)
    files = [f for f in files if os.path.isfile(os.path.join(overlaid_dir, f))]
    # 同じ症例内の番号を削除
    wsis = [f[:f.find('_')][:9] for f in files]
    # 重複を削除 
    wsis = natsorted(list(set(wsis)))
    random.seed(random_seed)
    random.shuffle(wsis)
    n = math.ceil(len(wsis) / cv)
    wsi_fold = [wsis[idx:idx + n] for idx in range(0, len(wsis), n)]

    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")

        wsi_fold_tmp = copy.deepcopy(wsi_fold)
        test_wsis_tmp = wsi_fold_tmp.pop(cv_num)
        trvl_wsis_tmp = sum(wsi_fold_tmp, [])

        train_wsis_tmp, valid_wsis_tmp = train_test_split(
            trvl_wsis_tmp, test_size=val_ratio, random_state=random_seed)

        train_wsis, valid_wsis, test_wsis = [], [], []
        for file in files:
            f = file.replace('_overlaid.tif', '')
            for wsi in train_wsis_tmp:
                if wsi in f:
                    train_wsis.append(f)
            for wsi in valid_wsis_tmp:
                if wsi in f:
                    valid_wsis.append(f)
            for wsi in test_wsis_tmp:
                if wsi in f:
                    test_wsis.append(f)
        
        train_wsis = natsorted(train_wsis)
        valid_wsis = natsorted(valid_wsis)
        test_wsis = natsorted(test_wsis)

        logging.info(f"[wsi]  train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(valid_wsis, output_dir + f"cv{cv_num}_valid_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== valid (wsi) ==\n")
            for i in range(len(valid_wsis)):
                f.write(f"{valid_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # imgs_dir = "/mnt/ssdsam/chemotherapy_strage/mnt2_LEV1/"
    output_dir = "/mnt/ssdsam/chemotherapy_strage/dataset/202112_chemotherapy/"

    # wsi_fold = [
    #     ["H19-12183_7", "H19-12183_8", "H19-12183_9", "H19-12183_10", "H19-12183_11", "H19-12183_13", "H19-00019_3", "H19-00019_4",  "H18-08754_9", "H18-08754_11", "H18-03929_4", "H18-03929_5"],
    #     ["H19-06473_2", "H19-06473_3", "H19-06473_4", "H19-06473_6", "H18-10055_5", "H18-10055_6", "H18-08203_8", "H18-08203_9", "H19-06584_3", "H19-06584_5", "H19-06584_6",],
    #     ["H19-06343_3", "H19-06343_4", "H19-06343_5", "H18-09611_8", "H18-09611_18",  "H18-05230_5", "H18-05230_8", "H19-09154_5", "H19-09154_6", "H19-09154_7", "H19-09154_8",],
    # ]

    cv = 3
    classes = [0, 1, 2]
    val_ratio = 0.2
    # save_dataset2(imgs_dir, output_dir, wsi_fold=wsi_fold, classes=classes, cv=cv, val_ratio=val_ratio)

    # save_dataset3(imgs_dir, output_dir, wsi_fold=wsi_fold, classes=classes, cv=cv)

    # save_dataset4(imgs_dir, output_dir, wsi_fold=wsi_fold_mixcases, classes=classes, cv=cv, val_ratio=val_ratio)

    overlaid_dir = "/mnt/ssdsam/chemotherapy_strage/mnt1/mask_cancergrade/overlaid_[0, 1, 2]/"
    save_dataset_chemo_wsi(overlaid_dir=overlaid_dir, output_dir=output_dir, cv=cv, val_ratio=val_ratio)
    # files = os.listdir(overlaid_dir)
    # files = [f for f in files if os.path.isfile(os.path.join(overlaid_dir, f))]
    # # 同じ症例内の番号を削除
    # wsis = [f[:f.find('_')][:9] for f in files]
    # # 重複を削除 
    # wsis = natsorted(list(set(wsis)))
    # random.seed(0)
    # random.shuffle(wsis)
    # n = math.ceil(len(wsis) / cv)
    # wsi_fold = [wsis[idx:idx + n] for idx in range(0, len(wsis), n)]
    # print(len(wsis))
