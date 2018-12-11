
import glob

import cv2
import numpy as np

import torch.utils.data as data
from sklearn.model_selection import KFold

from misc import utils

####
class SerialLoader(data.Dataset):
    def __init__(self, path_info_list, shape_augs=None, input_augs=None, run_mode='train'):
        self.path_info_list = path_info_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs
        self.run_mode = run_mode
        
    def __getitem__(self, idx):

        if self.run_mode == 'train':        
            path = self.path_info_list[idx]
        else:
            path, codename = self.path_info_list[idx]

        img_real = cv2.imread(path)
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)

        orig_size = np.array(img_real.shape[:2])

        # deterministic so it can be reused
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img_real = shape_augs.augment_image(img_real)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img_real = input_augs.augment_image(img_real)

        edge = utils.get_edge_map(img_real)
        input_img = edge[...,None] # HWC

        if self.run_mode == 'train':
            return input_img, img_real
        else:
            return input_img, codename, orig_size    

    def __len__(self):
        return len(self.path_info_list)

####
class ColonDataset(object):
    """
    Container for hard-coded dataset info
    """
    def __init__(self):
        src_dir = '' # ! Def here

        self.path_list = glob.glob('%s/*.jpg' % src_dir)
        self.path_list.sort() # to ensure same ordering across platform

        return

    def get_subset(self, nr_fold=5, fold_idx=None, mode=None):
        """

        if `k_fold` is an positive integer, will do nested k-fold split
        
        `fold_idx` must be supplied as a tuple (x, y) where x is the idx
        of the external k-fold and y is idx of the internal k-fold.

        `mode` has three possible value `train`, `valid` and `test`
        ---`test`  will return data from external k-fold with using `x` as idx
        ---`train` and `valid` will return data from the internal k-fold, 
            using both `x` and `y` as idx
        """
 
        train_fold = []
        valid_fold = []
        kf = KFold(n_splits=nr_fold, random_state=5, shuffle=True)
        for train_index, valid_index in kf.split(self.path_list):
            train_fold.append([self.path_list[idx] for idx in list(train_index)])
            valid_fold.append([self.path_list[idx] for idx in list(valid_index)])
        if mode == 'train' or mode == 'valid':
            path_list = train_fold[fold_idx[0]]
            sub_train_fold = []
            sub_valid_fold = []
            kf = KFold(n_splits=nr_fold, random_state=5, shuffle=True)
            for train_index, valid_index in kf.split(path_list):
                sub_train_fold.append([path_list[idx] for idx in list(train_index)])
                sub_valid_fold.append([path_list[idx] for idx in list(valid_index)])
            path_list = sub_train_fold[fold_idx[1]] if mode == 'train' \
                            else sub_valid_fold[fold_idx[1]]
        else:
            path_list = valid_fold[fold_idx[0]]

        return path_list
