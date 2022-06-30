# color correction and white balancing (20 points)

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import Imath
import OpenEXR
import csv

from cp_hw2 import read_colorchecker_gm, writeEXR
from utils import get_pts, save_pts, read_pts, read_exr_img, get_patch
from photographic_tonemapping import tone_mapping

image_type = ["tiff", "jpg"]
merging_type = ["linear", "logarithmic"]
weight_scheme = ["uniform", "tent", "gaussian", "photon"]
patch_type = list(range(1, 25))     # [1, 2, 3, ... , 24]

def compute_patch_mean(patches):
    patch_mean = {}
    for patch in patch_type:
        patch_mean[patch] = np.mean(patches[patch], (0, 1))
    return patch_mean

def get_gt_color():
    r, g, b = read_colorchecker_gm()
    gt_color = np.dstack([r, g, b])
    return gt_color

def get_color_checker(filename, origin_img_type):
    pts = read_pts("../result/pts_" + origin_img_type + "_all.csv", patch_type)
    _, patches = get_patch(pts, filename, patch_type)
    patches_mean = compute_patch_mean(patches)
    patches_mean = np.array(list(patches_mean.values()))
    patches_mean = np.reshape(patches_mean, (6, 4, 3)).transpose(1, 0, 2)
    return patches_mean

def checker2linear(color_checker):
    color_checker = color_checker.transpose(1, 0, 2).reshape(-1, 3)
    return color_checker

def color_affine_transform(color_checker, gt_color):
    color_checker = checker2linear(color_checker)
    gt_color = checker2linear(gt_color)
    A = np.hstack((color_checker, np.ones((color_checker.shape[0], 1))))
    b = gt_color
    transform_matrix, _, _, _ = np.linalg.lstsq(A, b)
    return transform_matrix

def color_correction(img, transform_matrix):
    correct_img = np.zeros(img.shape)
    img = np.dstack((img, np.ones((img.shape[0], img.shape[1], 1))))
    for i in range(correct_img.shape[1]):
        A = img[:, i, :].reshape(img.shape[0], img.shape[2])
        B = transform_matrix
        correct_img[:, i, :] = np.matmul(A, B)
    return correct_img

def white_balance(img, origin_img_type):
    pts = read_pts("../result/pts_" + origin_img_type + "_all.csv", patch_type)
    pt1 = list(map(int, eval(pts[4])[0]))
    pt2 = list(map(int, eval(pts[4])[1]))
    patch4 = img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
    patch4_mean = np.mean(patch4, (0, 1))
    gt_color = get_gt_color()
    gt_patch4 = checker2linear(gt_color)[3]
    k = gt_patch4 / patch4_mean
    img = img * k
    return img

def color_correction_white_balance(filename, img_type, weighting_scheme, merge_type, save_dir):
    img = read_exr_img(filename)
    color_checker = get_color_checker(filename, i)
    gt_color = get_gt_color()
    transform_matrix = color_affine_transform(color_checker, gt_color)
    color_corrected_img = color_correction(img, transform_matrix)
    cw_img = white_balance(color_corrected_img, i)
    img_name = "hdr_" + img_type + '_' + weighting_scheme + '_' + merge_type + "_ccwb.exr"
    writeEXR(save_dir + img_name, cw_img)

def draw_color_checker(filename, img_type):
    color_checker = get_color_checker(filename, img_type)
    gt_color = get_gt_color()
    plt.figure("color checker")
    plt.subplot(211)
    plt.title("average value")
    plt.imshow(color_checker / (1 + color_checker))
    plt.subplot(212)
    plt.title("ground truth")
    plt.imshow(gt_color)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # save point for each patch for jpg and tiff
    # for i in image_type:
    #     print("select points for " + i)
    #     pts = get_pts("../data/door_stack/exposure14." + i, patch_type)
    #     save_pts(pts, "../result/pts_" + i + "_all.csv", patch_type)

    # save a color corrected image
    i= image_type[1]
    w = weight_scheme[0]
    m = merging_type[1]
    filename = "/home/llipa/HDRimaging/result/hdr_" + i + '_' + w + '_' + m + ".exr"
    save_dir = "/home/llipa/HDRimaging/result/ccwb/"
    save_name = save_dir + "hdr_" + i + '_' + w + '_' + m + "_ccwb.exr"

    # draw color checker and standard RGB value
    # draw_color_checker(filename, i)

    color_correction_white_balance(filename, i, w, m, save_dir)

    img = tone_mapping(filename)
    cw_img = tone_mapping(save_name)
    
    plt.figure("color correction white balancing")
    plt.subplot(121)
    plt.title("origin HDR image")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(122)
    plt.title("color correction and white balancing")
    plt.imshow(cw_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("../result/ccwb/color_correction_white_balancing.png", bbox_inches = 'tight')
    plt.show()

