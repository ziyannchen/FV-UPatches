import numpy as np
import cv2


def pre_processing(data):
    # to enhance the vein
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)

    train_imgs = train_imgs / 255.

    return train_imgs.astype(np.float32)


def rgb2gray(rgb):
    # convert RGB image to gray image.
    r, g, b = rgb.split()
    return g


def clahe_equalized(images):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :], dtype = np.uint8))

    return images_equalized


def datasets_normalized(images):
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / images_std
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) / (np.max(images_normalized) - minv)) * 255
    
    return images_normalized


def adjust_gamma(images, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :], dtype = np.uint8), table)

    return new_images

