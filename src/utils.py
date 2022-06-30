# define some tool fuc

import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt
import csv
import OpenEXR
import Imath

from cp_hw2 import lRGB2XYZ

own_exposure_time = [0.125, 0.25, 0.5, 1., 2.]

def select_pixels(img_path, freq = 200):
    img = io.imread(img_path)
    return img[::freq, ::freq]

def get_exposure_time(num, is_own_data = False):
	exposure_time = (1 / 2048) * np.exp2(num)
	if is_own_data:
		exposure_time = own_exposure_time[num]
	return exposure_time

def uniform_weights(img_pixels, linearization):
	if isinstance(img_pixels, np.ndarray):
		if linearization == False:
			weights = np.ones(img_pixels.shape)
			lower_mask = np.where(img_pixels < 0.05)
			upper_mask = np.where(img_pixels > 0.95)
		else:
			weights = np.ones(img_pixels.shape) * 255
			lower_mask = np.where(img_pixels < 0)
			upper_mask = np.where(img_pixels > 255)
		weights[lower_mask] = 0
		weights[upper_mask] = 0
		return weights
	else:
		if linearization == False:
			if img_pixels < 0.05 or img_pixels > 0.95:
				return 0
			else:
				return 1
		else:
			if img_pixels < 0 or img_pixels > 255:
				return 0
			else:
				return 255

def tent_weights(img_pixels, linearization):
	if isinstance(img_pixels, np.ndarray):
		if linearization == False:
			weights = np.minimum(img_pixels, 1 - img_pixels)
			lower_mask = np.where(img_pixels < 0.05)
			upper_mask = np.where(img_pixels > 0.95)
		else:
			weights = np.minimum(img_pixels, 255 - img_pixels)
			lower_mask = np.where(img_pixels < 0)
			upper_mask = np.where(img_pixels > 255)
		weights[lower_mask] = 0
		weights[upper_mask] = 0
		return weights
	else:
		if linearization == False:
			if img_pixels < 0.05 or img_pixels > 0.95:
				return 0
			else:
				return min(img_pixels, 1 - img_pixels)
		else:
			if img_pixels < 0 or img_pixels > 255:
				return 0
			else:
				return min(img_pixels, 255 - img_pixels)

def gaussian_weights(img_pixels, linearization):
	if isinstance(img_pixels, np.ndarray):
		if linearization == False:
			weights = np.exp(-4 * (img_pixels - 0.5)**2 / (0.5)**2)
			lower_mask = np.where(img_pixels < 0.05)
			upper_mask = np.where(img_pixels > 0.95)
		else:
			weights = np.exp(-4 * (img_pixels / 255 - 0.5)**2 / (0.5)**2) * 255
			lower_mask = np.where(img_pixels < 0)
			upper_mask = np.where(img_pixels > 255)
		weights[lower_mask] = 0
		weights[upper_mask] = 0
		return weights
	else:
		if linearization == False:
			if img_pixels < 0.05 or img_pixels > 0.95:
				return 0
			else:
				return np.exp(-4 * (img_pixels - 0.5)**2 / (0.5)**2)
		else:
			if img_pixels < 0 or img_pixels > 255:
				return 0
			else:
				return np.exp(-4 * (img_pixels / 255 - 0.5)**2 / (0.5)**2) * 255

def photon_weights(img_pixels, exposure_time, linearization):
	if isinstance(img_pixels, np.ndarray):
		if linearization == False:
			weights = np.ones(img_pixels.shape) * exposure_time
			lower_mask = np.where(img_pixels < 0.05)
			upper_mask = np.where(img_pixels > 0.95)
		else:
			weights = np.ones(img_pixels.shape) * exposure_time * 255
			lower_mask = np.where(img_pixels < 0)
			upper_mask = np.where(img_pixels > 255)
		weights[lower_mask] = 0
		weights[upper_mask] = 0
		return weights
	else:
		if linearization == False:
			return 1
		else:
			return 255

def get_weights(img_pixels, weighting_scheme, exposure_time = 0, linearization = False):
    if weighting_scheme == 'uniform':
        return uniform_weights(img_pixels, linearization)
    elif weighting_scheme == 'tent':
        return tent_weights(img_pixels, linearization)
    elif weighting_scheme == 'gaussian':
        return gaussian_weights(img_pixels, linearization)
    elif weighting_scheme == 'photon':
        return photon_weights(img_pixels, exposure_time, linearization)
    else:
        print("illegal weighting scheme: " + weighting_scheme + '.\n')
        assert(0)

def get_pts(img_path, patch_type):
    img = io.imread(img_path)
    pts = {}
    for patch in patch_type:
        print("select points for patch " + str(patch))
        plt.imshow(img)
        patch_pt = plt.ginput(n = 2)
        plt.close()
        pts[patch] = patch_pt
    return pts

def save_pts(pts, save_file, patch_type):
    with open(save_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=patch_type)
        writer.writerow(pts)

def read_pts(filename, patch_type):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=patch_type)
        for r in reader:
            pts = r
    return pts

def read_exr_img(filename):
    hdr_image = OpenEXR.InputFile(filename)
    dw = hdr_image.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.frombuffer(c, np.float32).reshape(size) for c in hdr_image.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
    img = np.dstack(data)
    return img

def get_patch(pts, filename, patch_type):
    img = read_exr_img(filename)
    lin_img = lRGB2XYZ(img)[:, :, 1]
    patches_Y = {}
    patches_RGB = {}
    for patch in patch_type:
        pt1 = list(map(int, eval(pts[patch])[0]))
        pt2 = list(map(int, eval(pts[patch])[1]))
        p_Y = lin_img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        p_RGB = img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        # show where is the patches
        # origin_img = io.imread("../data/door_stack/exposure14.tiff")
        # origin_img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1] = 256 - origin_img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        # plt.imshow(origin_img)
        # plt.show()
        patches_Y[patch] = p_Y
        patches_RGB[patch] = p_RGB
    return patches_Y, patches_RGB