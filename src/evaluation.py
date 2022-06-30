# Evaluation (10 points)

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import Imath
import OpenEXR
import csv

from utils import get_pts, save_pts, read_pts, read_exr_img, get_patch

image_type = ["tiff", "jpg"]
merging_type = ["linear", "logarithmic"]
weight_scheme = ["uniform", "tent", "gaussian", "photon"]
patch_type = [4, 8, 12, 16, 20, 24]

def compute_patch_log(patches):
    patch_log = {}
    for patch in patch_type:
        patch_log[patch] = np.log(np.mean(patches[patch]))
    return patch_log

def evaluate_patch(patch_log):
    A = np.vstack([patch_type, np.ones(len(patch_type))]).T
    par, res, _, _ = np.linalg.lstsq(A, np.array(list(patch_log.values())))
    return par, res

def evaluate_hdr(filename, origin_img_type):
    pts = read_pts("../result/pts_" + origin_img_type + ".csv", patch_type)
    patches, _ = get_patch(pts, filename, patch_type)
    patch_log = compute_patch_log(patches)
    # draw patch log
    plt.plot(patch_type, list(patch_log.values()), label = filename.split("hdr_")[1].split(".ex")[0])
    par, res = evaluate_patch(patch_log)
    return par, res

def evaluate_all_hdr():
    eval_results = []
    for i in image_type:
        for w in weight_scheme:
            for m in merging_type:
                filename = "/home/HDRimaging/assgn2/result/hdr_" + i + '_' + w + '_' + m + ".exr"
                par, res = evaluate_hdr(filename, i)
                eval_results.append([i, w, m, par, res])
    return eval_results

if __name__ == "__main__":
    # save point for each patch for jpg and tiff
    # for i in image_type:
    #     print("select points for " + i)
    #     pts = get_pts("../data/door_stack/exposure14." + i, patch_type)
    #     save_pts(pts, "../result/pts_" + i + ".csv", patch_type)

    # evaluate all exr images and save the results
    plt.figure("log value")
    eval_results = evaluate_all_hdr()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../result/log_value.png", bbox_inches = 'tight')
    plt.show()
    with open("../result/evaluate.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(eval_results)
