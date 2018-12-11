
import os
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
from progress.bar import Bar as ProgressBar

from .utils import get_stats_2d_array 

####
def compute_stat_patch(src_img_path, rec_img_path, ann_id, idx):
    """
    Expect each patch has a corresponding label for one type, not pixel-wise annotation
    """

    rgb_src_img = cv2.imread(src_img_path)
    rgb_src_img = cv2.cvtColor(rgb_src_img, cv2.COLOR_BGR2RGB) 

    rgb_rec_img = cv2.imread(rec_img_path)
    rgb_rec_img = cv2.cvtColor(rgb_rec_img, cv2.COLOR_BGR2RGB)

    gray_src_img = cv2.cvtColor(rgb_src_img, cv2.COLOR_RGB2GRAY)
    gray_rec_img = cv2.cvtColor(rgb_rec_img, cv2.COLOR_RGB2GRAY)

    gray_stats = get_stats_2d_array(gray_src_img, gray_rec_img, roi=None)
    # for RGB, calculate per channel then taking average
    rgb_stats = []
    for ch_idx in range (0, 3):
        ch_src_img = rgb_src_img[...,ch_idx]
        ch_rec_img = rgb_rec_img[...,ch_idx]
        rgb_ch_stats = get_stats_2d_array(ch_src_img, ch_rec_img, roi=None)
        rgb_stats.append(rgb_ch_stats)
    rgb_stats = np.array(rgb_stats)
    rgb_stats = np.mean(rgb_stats, axis=0)
    rgb_stats = list(rgb_stats)

    result = [rgb_stats, gray_stats, ann_id, idx]
    return result
    
####
def run(src_file_list, rec_file_list, output_path, 
        foc_file_list=None, encode_label=None):
    """
    Args:
        encode_label: set it to some non-neg integer X to assume all patch within 
                      `src_file_list` having label X, else it will decode the
                      filename to get the label asssuming the form "*_X.(jpg|png|etc.)" 
    """
    nr_procs = 0
    ############
    file_info_list = []
    for idx, src_file_path in enumerate(src_file_list):
        filename = os.path.basename(src_file_path)
        basename = filename.split('.')[0]
        label_id = encode_label if encode_label is not None \
                                else int(basename.split('_')[-1])
        if foc_file_list is None:
            file_info_list.append([src_file_path, rec_file_list[idx], 
                                   label_id, idx])
        else:
            file_info_list.append([src_file_path, rec_file_list[idx], 
                                   label_id, idx, foc_file_list[idx]])

    output_path_info = Path(output_path)
    if not os.path.isdir(output_path_info.parent):
        os.makedirs(output_path_info.parent)

    stat_list = []
    pbar = ProgressBar('Processing', max=len(file_info_list), width=48)
    if nr_procs > 1:
        def proc_callback(x):
            stat_list.append(x)
            pbar.next()

        with Pool(processes=nr_procs) as proc_pool:
            result_obj_list = []
            for file_info in file_info_list:
                # a handler for extracting actual result, not the function's returned output
                result_obj = proc_pool.apply_async(compute_stat_patch, 
                                        file_info, callback=proc_callback)
                result_obj_list.append(result_obj)
            for result_obj in result_obj_list:
                result_obj.wait()
    else:
        for file_info in file_info_list:
            stat_list.append(compute_stat_patch(*file_info))
            pbar.next()
    pbar.finish()

    # remove empty result due to no rois
    stat_list = np.array(stat_list)
    np.save(output_path, stat_list)
