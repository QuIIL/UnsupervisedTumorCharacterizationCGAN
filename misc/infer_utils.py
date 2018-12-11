
import os

import glob
import cv2
import numpy as np

from misc import utils

####
def get_patches_stat(img, win_size):
    if img.shape[0] % win_size[0] != 0:
        nr_row_patches = (int(img.shape[0] / win_size[0]) + 1)
    else:
        nr_row_patches = int(img.shape[0] / win_size[0])

    if img.shape[1] % win_size[1] != 0:
        nr_col_patches = (int(img.shape[1] / win_size[1]) + 1)
    else:
        nr_col_patches = int(img.shape[1] / win_size[1])
    return nr_row_patches, nr_col_patches
####
def get_patches_list(img, win_size):
    # ! no overlapping only
    img = np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img
    nr_row_patches = int(img.shape[0] / win_size[0])
    nr_col_patches = int(img.shape[1] / win_size[1])
    patches = np.reshape(img, (nr_row_patches, win_size[0], nr_col_patches, win_size[1], img.shape[-1]))
    patches = np.transpose(patches, (0, 2, 1, 3, 4))
    patches = np.reshape(patches, (nr_row_patches * nr_col_patches, win_size[0], win_size[1], img.shape[-1]))
    return list(patches)

####
def create_cache_patch(path_list, cache_dir, patch_size):
    """
    Only support square patch
    """
    assert isinstance(patch_size, int), 'Only support square patch'
    for path_idx, path in enumerate(path_list):
        img = cv2.imread(path)
        img = utils.center_pad_to_div(img, patch_size)
        img_patches = get_patches_list(img, (patch_size, patch_size))
        for _, patch in enumerate(img_patches):
            save_path = '%s/%d_%d.jpg' % (cache_dir, path_idx, patch)
            cv2.imwrite(save_path,  patch)            
    return
####
def assemble_cache_patch(src_path_list, cache_dir, output_dir, patch_size, ext='.jpg'):
    """
    If the `path_list` is in the same order as when inputted to 
    `create_cache_patch`, the assembled will be in the same order
    """
    path_list = glob.glob('%s/*%s' % (cache_dir, ext))
    path_list.sort()
    basename_list = [os.path.basename(path) for path in path_list]
    src_idx_list = [name.split('.')[0].split('_')[0] for name in basename_list]
    src_idx_list = list(set(src_idx_list))

    for src_idx in src_idx_list:
        patch_name_list = [name for name in basename_list if src_idx in name]
        patch_name_list.sort()

        src_img = cv2.imread(src_path_list[int(src_idx)])
        nr_row_patches, nr_col_patches = get_patches_stat(src_img, (patch_size, patch_size))

        patch_list = [cv2.imread('%s/%s' % (cache_dir, name)) for name in patch_name_list]

        img_recon = np.array(patch_list)
        img_recon = np.reshape(img_recon, (nr_row_patches, nr_col_patches, patch_size, patch_size, 3))
        img_recon = np.transpose(img_recon, (0, 2, 1, 3, 4))
        img_recon = np.reshape(img_recon, (nr_row_patches * patch_size, nr_col_patches * patch_size, 3))
        img_recon = utils.cropping_center(img_recon, src_img.shape[:2])
        cv2.imwrite('%s/%s.jpg' % (output_dir, src_idx), img_recon)
    return
####
