# Create and tonemap your own HDR photo (50 points)

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from merge_HDR import merge_hdr
from photographic_tonemapping import tone_mapping

image_type = ["tiff", "jpg"]
merging_type = ["linear", "logarithmic"]
weight_scheme = ["uniform", "tent", "gaussian", "photon"]


if __name__ == "__main__":
    save_dir = ["/home/llipa/HDRimaging/result_own/image_set1/", "/home/llipa/HDRimaging/result_own/image_set2/"]

    # merge hdr for 16 types for 2 dataset
    # for num, data in enumerate(["../data/own/image_set1/", "../data/own/image_set2/"]):
    #     for i in image_type:
    #         for w in weight_scheme:
    #             merge_hdr(data, i, w, False, 5, save_dir[num], True)
    #             merge_hdr(data, i, w, True, 5, save_dir[num], True)


    # show HDR img
    i= image_type[1]
    w = weight_scheme[0]
    m = merging_type[1]
    filename = save_dir[1] + "hdr_" + i + '_' + w + '_' + m + ".exr"
    savename = save_dir[1] + "hdr_" + i + '_' + w + '_' + m + ".png"

    img_RGB = tone_mapping(filename)
    img_Y = tone_mapping(filename, color_space='Y')

    plt.figure("own data")
    plt.subplot(121)
    plt.title("RGB")
    plt.imshow(img_RGB)
    plt.axis('off')
    plt.subplot(122)
    plt.title("Y")
    plt.imshow(img_Y)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savename, bbox_inches = 'tight')
    plt.show()
