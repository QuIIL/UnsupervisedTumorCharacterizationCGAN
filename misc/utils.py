
import os
import shutil

import cv2
import numpy as np

####
def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
####
def cropping_center(x, crop_shape, batch=False):   
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]        
    return x
####
def checkerboard_masking(img, box, cell_size, color=255, inverse=False):
    """
    TODO: optimize this
    """
    out_img = np.copy(img)

    nr_cell_r = int(box[0] / cell_size[0]  + 0.5)
    nr_cell_c = int(box[1] / cell_size[1]  + 0.5)
    nr_cell_r =  nr_cell_r + nr_cell_r % 2
    nr_cell_c =  nr_cell_c + nr_cell_c % 2
    board = np.array([[1, 0] * (nr_cell_c // 2),
                      [0, 1] * (nr_cell_c // 2)] 
                    * (nr_cell_r // 2))
    board = np.kron(board, np.ones(cell_size))
    board = cropping_center(board, box)

    h, w = out_img.shape[:2]
    orig_y = h // 2
    orig_x = w // 2
    place = out_img[orig_y - box[0] // 2 : orig_y + box[0] // 2,
                    orig_x - box[1] // 2 : orig_x + box[1] // 2]
    if inverse:
        place[board == 0] = color
    else:
        place[board == 1] = color
    return out_img
####
def get_edge_map(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edge_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    edge_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    edge = np.hypot(edge_x, edge_y) # normalize (Q&D)
    edge = ((edge / (np.max(edge)+ 1.0e-8)) * 255.0).astype('uint8')  
    return edge
####
def color_mask(a, r, g, b):
    ch_r = a[...,0] == r
    ch_g = a[...,1] == g
    ch_b = a[...,2] == b
    return ch_r & ch_g & ch_b
####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
####
def center_pad_to_div(img, div, cval=255):
    # rounding down, add 1
    div_h = int(img.shape[0] / div) + 1
    div_w = int(img.shape[1] / div) + 1
    pad_h = div_h * div - img.shape[0]
    pad_w = div_w * div - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    pad = (pad_h, pad_w) if len(img.shape) == 2 \
            else (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad, 'constant', constant_values=cval)
    return img
####
def center_pad_to_shape(img, size, cval=255):
    # ! shape assertion
    # rounding down, add 1
    pad_h = size - img.shape[0]
    pad_w = size - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, 'constant', constant_values=cval)
    return img
####
def rotate(mat, angle, bound=False):
    """
    Rotates an image (uint8) by angle (in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    if not bound:
        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (width, height))
    else:
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat
####
