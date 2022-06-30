# Linearize rendered images (20 points)

import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt

from utils import get_exposure_time
from utils import select_pixels
from utils import get_weights

def init_matrix(img_path, num, weighting_scheme, is_own):
    # select N points for optimization
    img_pixels = select_pixels(img_path)
    # get exposure time for this img
    exposure_time = get_exposure_time(num, is_own)
    # remove bad points
    img_pixels[np.where(img_pixels > 255)] = 255
    img_pixels[np.where(img_pixels < 0)] = 0
    # single g for all the channels
    img_pixels = np.reshape(img_pixels, (-1, 1))
    # init weights
    weights = get_weights(img_pixels, weighting_scheme, exposure_time, linearization = True)

    A = np.zeros((0,0))
    b = []
    for index in range(img_pixels.shape[0]):
        # A
        color_row = np.zeros((1,256))
        color_row[0, img_pixels[index, 0]] = weights[index, 0]
        radiance_row = np.zeros((1, img_pixels.shape[0]))
        radiance_row[0, index] = -weights[index, 0]
        pixel_row = np.hstack((color_row, radiance_row))
        if A.shape[0] == 0:
            A = pixel_row
        else:
            A = np.vstack((A, pixel_row))
        # b
        b_entry = weights[index, 0] * np.log(exposure_time)
        b.append(b_entry)
    b = np.asarray(b)
    b = np.reshape(b, (-1, 1))
    return A, b


def linearize(img_dir, weighting_scheme, num_img, is_own = False):
    A = np.zeros((0,0))
    b = np.zeros((0,0))
    # A matrix value from images
    for i in range(num_img):
        img_path = img_dir + "exposure" + str(i + 1) + ".jpg"
        A_mat, b_vec = init_matrix(img_path, i, weighting_scheme, is_own)
        if A.shape[0] == 0:
            A = A_mat
            b = b_vec
        else:
            A = np.vstack((A, A_mat))
            b = np.vstack((b, b_vec))
    # A matrix value from smooth regularization
    laplacian_submatrix = np.zeros((254, A.shape[1]))
    b_submatrix = np.zeros((254, 1))
    lambda_val = 10
    for i in range(254):
        index = i + 1
        laplacian_submatrix[i, index] = -2 * get_weights(index, weighting_scheme, linearization = True) * lambda_val
        laplacian_submatrix[i, index - 1] = get_weights(index, weighting_scheme, linearization = True) * lambda_val
        laplacian_submatrix[i, index + 1] = get_weights(index, weighting_scheme, linearization = True) * lambda_val

    # A
    A_extra_row = np.ones((1, A.shape[1]))
    A = np.vstack((A, A_extra_row))
    A = np.vstack((A, laplacian_submatrix))
    # b
    b = np.vstack((b, np.array([[1]])))
    b = np.vstack((b, b_submatrix))\
    # g
    g = np.linalg.lstsq(A, b)
    g = g[0]
    g = g[:256]
    g = np.reshape(g, (-1,))
    return g



if __name__ == "__main__":
    # plot g value
    g_uniform = linearize("../data/door_stack/", "uniform", 16)
    g_tent = linearize("../data/door_stack/", "tent", 16)
    g_gaussian = linearize("../data/door_stack/", "gaussian", 16)
    g_photon = linearize("../data/door_stack/", "photon", 16)
    plt.figure("g value")
    plt.subplot(221)
    plt.title("uniform")
    plt.ylabel("g")
    plt.plot(g_uniform, '.')
    plt.subplot(222)
    plt.title("tent")
    plt.plot(g_tent, '.')
    plt.subplot(223)
    plt.title("gaussian")
    plt.xlabel("pixel value")
    plt.ylabel("g")
    plt.plot(g_gaussian, '.')
    plt.subplot(224)
    plt.title("photon")
    plt.xlabel("pixel value")
    plt.plot(g_photon, '.')
    plt.tight_layout()
    plt.savefig("../result/g_value.png", bbox_inches = 'tight')
    plt.show()