# Merge exposure stack into HDR image (25 points)

import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt
from linearize import linearize

from utils import get_exposure_time, get_weights
from cp_hw2 import writeEXR

merging_type = ["linear", "logarithmic"]
weight_scheme = ["uniform", "tent", "gaussian", "photon"]
image_type = ["tiff", "jpg"]

def merge_hdr(img_dir, img_type, weighting_scheme, islog, num_img, save_dir, is_own = False):
    numerator = 0
    denominator = 0
    g = linearize(img_dir, weighting_scheme, num_img, is_own)
    for i in range(num_img):
        if img_type == "tiff":
            img_path = img_dir + "exposure" + str(i + 1) + ".tiff"
            img = io.imread(img_path)
            ldr_img = img
            lin_img = img
        elif img_type == "jpg":
            img_path = img_dir + "exposure" + str(i + 1) + ".jpg"
            img = io.imread(img_path)
            ldr_img = img
            # linearize
            lin_img = np.exp(g[img])
        exposure_time = get_exposure_time(i, is_own)
        ldr_weights = get_weights(ldr_img / 255, weighting_scheme, exposure_time)
        if islog:
            numerator += ldr_weights * (np.log(lin_img + 0.0001) - np.log(exposure_time))
        else:
            numerator += ldr_weights * lin_img / exposure_time
        denominator += ldr_weights
    
    zero_mask = denominator == 0
    upper_mask = np.logical_and(zero_mask, img > 128)
    lower_mask = np.logical_and(zero_mask, img < 128)
    denominator[zero_mask] = 1
    if islog:
        hdr_img = np.exp(numerator / denominator)
    else:
        hdr_img = numerator / denominator
    hdr_img[upper_mask] = np.max(hdr_img)
    hdr_img[lower_mask] = np.min(hdr_img)
    img_name = "hdr_" + img_type + '_' + weighting_scheme + '_' + merging_type[islog] + ".exr"
    writeEXR(save_dir + img_name, hdr_img)
    return img_name, hdr_img



if __name__ == "__main__":
    # merge hdr for 16 types
    for i in image_type:
        for w in weight_scheme:
            merge_hdr("../data/door_stack/", i, w, False, 16, "/home/llipa/HDRimaging/result/")
            merge_hdr("../data/door_stack/", i, w, True, 16, "/home/llipa/HDRimaging/result/")