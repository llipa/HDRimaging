# Photographic tonemapping (20 points)

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import Imath
import OpenEXR
import copy

from cp_hw2 import lRGB2XYZ, XYZ2lRGB, xyY_to_XYZ
from utils import read_exr_img

image_type = ["tiff", "jpg"]
merging_type = ["linear", "logarithmic"]
weight_scheme = ["uniform", "tent", "gaussian", "photon"]

def gamma_encoding(im) :
    im_rgb = copy.copy(im)
    im1 = im_rgb[:,:,0]
    im2 = im_rgb[:,:,1]
    im3 = im_rgb[:,:,2]
    imred_lower_bound = im_rgb[:, :, 0] < 0.0031308
    imred_upper_bound = im_rgb[:, :, 0] >= 0.0031308

    imgreen_lower_bound = im_rgb[:, :, 1] < 0.0031308
    imgreen_upper_bound = im_rgb[:, :, 1] >= 0.0031308

    imblue_lower_bound = im_rgb[:, :, 2] < 0.0031308
    imblue_upper_bound = im_rgb[:, :, 2] >= 0.0031308

    im1[imred_lower_bound] = 12.92 * im1[imred_lower_bound]
    im1[imred_upper_bound] = (1 + 0.055) * (im1[imred_upper_bound]) ** (1 / 2.4) - 0.055

    im2[imgreen_lower_bound] = 12.92 * im2[imgreen_lower_bound]
    im2[imgreen_upper_bound] = (1 + 0.055) * (im2[imgreen_upper_bound]) ** (1 / 2.4) - 0.055

    im3[imblue_lower_bound] = 12.92 * im3[imblue_lower_bound]
    im3[imblue_upper_bound] = (1 + 0.055) * (im3[imblue_upper_bound]) ** (1/2.4) - 0.055

    im_gamma = np.dstack((im1, im2, im3))
    return im_gamma

def tone_mapping(hdr_path, K = 0.15, B = 0.95, color_space = 'RGB') :
    save_path = hdr_path.split('.')
    save_name = save_path[0] + '_' + str(K) + '_' +  str(B) + '_' +  color_space + '.png'
    img = read_exr_img(hdr_path)
    img = np.clip(img, 0, None)
    if color_space == 'RGB':
        img_tonemapped = np.zeros(np.shape(img))
        im_hdr = np.exp(np.mean(np.log(img + 0.0001)))
        for channel in range(3):
            im_tilda_hdr = (K / im_hdr) * img[:, :, channel]
            im_tilda_white = B * np.max(im_tilda_hdr)
            im_tonemapped_channel = (im_tilda_hdr * (1 + im_tilda_hdr / (im_tilda_white * im_tilda_white))) / (1 + im_tilda_hdr)
            img_tonemapped[:, :, channel] = im_tonemapped_channel
        im_gamma_correct = gamma_encoding(img_tonemapped)
        # plt.imshow(im_gamma_correct)
        # plt.savefig(save_name)
        return im_gamma_correct
    elif color_space == 'Y':
        XYZ_im = lRGB2XYZ(img)
        x_channel = XYZ_im[:, :, 0] / (XYZ_im[:, :, 0] + XYZ_im[:, :, 1] + XYZ_im[:, :, 2])
        y_channel = XYZ_im[:, :, 1] / (XYZ_im[:, :, 0] + XYZ_im[:, :, 1] + XYZ_im[:, :, 2])
        Y_image = XYZ_im[:, :, 1]
        im_hdr = np.exp(np.mean(np.log(Y_image + 0.0001)))
        im_tilda_hdr = (K / im_hdr) * Y_image
        im_tilda_white = B * np.max(im_tilda_hdr)
        im_tonemapped = (im_tilda_hdr * (1 + im_tilda_hdr / (im_tilda_white * im_tilda_white))) / (1 + im_tilda_hdr)
        Y_channel = im_tonemapped
        X_final, Y_final, Z_final = xyY_to_XYZ(x_channel, y_channel, Y_channel)
        XYZ_final = np.dstack((X_final, Y_final, Z_final))
        im_rgb = XYZ2lRGB(XYZ_final)
        im_gamma_correct = gamma_encoding(im_rgb)
        # plt.imshow(im_gamma_correct)
        # plt.savefig(save_name)
        return im_gamma_correct


if __name__ == "__main__":
    # show a exr image
    i= image_type[1]
    w = weight_scheme[0]
    m = merging_type[1]
    ccwb_filename = "/home/llipa/HDRimaging/result/ccwb/hdr_" + i + '_' + w + '_' + m + "_ccwb.exr"

    # change K
    plt.figure("photographic tonemapping K")
    num = 0
    num_subplot = [231, 234, 232, 235, 233, 236]
    for K in [0.15, 0.3, 0.45]:
        for B in [0.95]:
            img_RGB = tone_mapping(ccwb_filename, K, B)
            plt.subplot(num_subplot[num])
            plt.title("RGB" + " K" + str(K) + " B" + str(B))
            plt.imshow(img_RGB)
            plt.axis('off')
            num += 1
            img_Y = tone_mapping(ccwb_filename, K, B, color_space='Y')
            plt.subplot(num_subplot[num])
            plt.title("Y" + " K" + str(K) + " B" + str(B))
            plt.imshow(img_Y)
            plt.axis('off')
            num += 1
    
    plt.tight_layout()
    plt.savefig("../result/ccwb/photographic_tonemapping_K.png", bbox_inches = 'tight')
    plt.show()


    # change B
    plt.figure("photographic tonemapping B")
    num = 0
    num_subplot = [231, 234, 232, 235, 233, 236]
    for K in [0.15]:
        for B in [0.95, 0.80, 0.65]:
            img_RGB = tone_mapping(ccwb_filename, K, B)
            plt.subplot(num_subplot[num])
            plt.title("RGB" + " K" + str(K) + " B" + str(B))
            plt.imshow(img_RGB)
            plt.axis('off')
            num += 1
            img_Y = tone_mapping(ccwb_filename, K, B, color_space='Y')
            plt.subplot(num_subplot[num])
            plt.title("Y" + " K" + str(K) + " B" + str(B))
            plt.imshow(img_Y)
            plt.axis('off')
            num += 1
    
    plt.tight_layout()
    plt.savefig("../result/ccwb/photographic_tonemapping_B.png", bbox_inches = 'tight')
    plt.show()