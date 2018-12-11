
import math
import numpy as np

import cv2
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from skimage.measure import compare_ssim
from skimage.util.arraycrop import crop

import pandas as pd
import scipy.stats as stats
from scipy.ndimage import measurements
from skimage.morphology import (binary_dilation, binary_erosion,
                                remove_small_holes,
                                remove_small_objects)

####
def corr_coeff(a, b):
    """
    matlab corr2 for 2-D array, match with np.corrcoeff,
    np.corrcoeff return symmetrical 2-d array matrix for a, b (position table)
    corr_coeff return single value
    """
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum() + 1.0e-6)
    return r

####
def mutual_info(a, b):
    "to check logic of sklearn code, both matches"
    indrow_a = a.flatten()
    indcol_b = b.flatten()
    # at position x, y within image a (having intensity i) and b
    # having intensity j), increase the counter at i, j for histogram
    joint_hist = np.histogram2d(indcol_b, indrow_a, bins=256)[0]
    joint_prob = joint_hist / indrow_a.size
    joint_prob_non_zero = joint_prob[joint_hist != 0]
    joint_entropy = -np.sum(joint_prob_non_zero * np.log(joint_prob_non_zero))
    # only non-zero elements
    a_hist = np.histogram(indrow_a, bins=256)[0]
    a_non_zero_hist = np.array(a_hist[a_hist !=0])
    a_non_zero_prob = a_non_zero_hist / np.sum(a_non_zero_hist)
    a_entropy = -np.sum(a_non_zero_prob * np.log(a_non_zero_prob))
    #
    b_hist = np.histogram(indcol_b, bins=256)[0]
    b_non_zero_hist = np.array(b_hist[b_hist !=0])
    b_non_zero_prob = b_non_zero_hist / np.sum(b_non_zero_hist)
    b_entropy = -np.sum(b_non_zero_prob * np.log(b_non_zero_prob))
    #
    mutual_information = a_entropy + b_entropy - joint_entropy
    return mutual_information

####
def get_stats_2d_array(src_img, rec_img, roi=None):
    src_img_flat = src_img.flatten()
    rec_img_flat = rec_img.flatten()
    if roi is None:
        arr_corr_coeff = corr_coeff(src_img_flat, rec_img_flat)
        arr_mutual_info = mutual_info_score(src_img_flat, rec_img_flat)
        arr_struct_sim = compare_ssim(src_img, rec_img)
    else:
        assert roi.dtype == np.bool, 'RoI must be boolean array so that it can be used to do numpy selection.'
        roi_flat = roi.flatten()
        if roi_flat.sum() == 0: # empty RoI
            arr_corr_coeff  = -1
            arr_mutual_info = -1
            arr_struct_sim  = -1
        src_img_flat = src_img_flat[roi_flat]
        rec_img_flat = rec_img_flat[roi_flat]
        arr_corr_coeff = corr_coeff(src_img_flat, rec_img_flat)
        arr_mutual_info = mutual_info_score(src_img_flat, rec_img_flat)
        # * get the full SIM image, then select the RoI, manually crop
        # * the border beforehand to match with sklearn, may need to check
        # * again with sklearn later if they decide to change the window size
        arr_struct_sim = compare_ssim(src_img, rec_img, full=True)[-1]
        arr_struct_sim = np.mean(crop(arr_struct_sim, 3)[crop(roi, 3)])
    arr_corr_coeff  = arr_corr_coeff if not np.isnan(arr_corr_coeff) else 0
    arr_mutual_info = arr_mutual_info if not np.isnan(arr_mutual_info) else 0
    arr_struct_sim = arr_struct_sim if not np.isnan(arr_struct_sim) else 0
    return [arr_corr_coeff, arr_mutual_info, arr_struct_sim]
####
def exclude_white_area(src_gray, src_ann, factor=1):    
    img_shape = src_ann.shape
    # * threshold at lower resolution can avoid the noisy artifacts within stroma
    src_gray = cv2.resize(src_gray, (0, 0), fx=1/4*factor, fy=1/4*factor)
    thmap = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thmap = np.array(thmap == 0, dtype=np.uint8)
    thmap = binary_dilation(thmap)
    thmap = binary_dilation(thmap)
    thmap = binary_erosion(thmap)
    thmap = np.array(thmap == 0, dtype=np.uint8)
    thmap = measurements.label(thmap)[0]
    thmap = remove_small_objects(thmap, min_size=(50/factor) * (50/factor))
    # * dont use binary_fill_holes, it can fill everything !
    thmap = measurements.label(thmap == 0)[0]
    thmap = remove_small_objects(thmap, min_size=(50/factor) * (50/factor))
    thmap = np.array(thmap == 0, dtype=np.uint8)
    # * cv2 resize requires width, numpy layout is height width !
    thmap = cv2.resize(thmap, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    src_ann[thmap > 0] = 0
    return src_ann
####
def anova_oneway(df):
    # * One-way Anova by hands, match F-test and p with scipy
    N = len(df.values)
    k = len(pd.unique(df.Labels))
    n = df.groupby('Labels').size().values

    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1
    SSbetween = (sum(df.groupby('Labels').sum()['Measurement'].values ** 2 / n)) \
              - (df['Measurement'].sum()**2) / N
    sum_y_squared = sum([value**2 for value in df['Measurement'].values])
    SSwithin = sum_y_squared - sum(df.groupby('Labels').sum()['Measurement']**2 / n) 
    SStotal = sum_y_squared - (df['Measurement'].sum()**2) / N
    MSbetween = SSbetween / DFbetween
    MSwithin = SSwithin / DFwithin
    F = MSbetween / MSwithin
    p = stats.f.sf(F, DFbetween, DFwithin)
    eta_sqrd = SSbetween / SStotal
    om_sqrd = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)

    return F, np.log10(p), eta_sqrd, om_sqrd
####