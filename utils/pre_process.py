import numpy as np
from PIL import Image
import cv2

def pre_processing(data, return_dtype=float):
    
    '''
    Image enhancement
    input: data: [0, 255]
    output: 
    '''
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    data = data.astype(np.float32)
    img = datasets_normalized(data)
    img = clahe_equalized(img)
    img = adjust_gamma(img, 1.2)
    
    img = img / 255.
    return img.astype(return_dtype)


# convert RGB image to gray image.
def rgb2gray(rgb):
    r, g, b = rgb.split()
    return g


def clahe_equalized(images):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :],
                                                            dtype = np.uint8))

    return images_equalized


def datasets_normalized(images):

    images_normalized = np.empty(images.shape)
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / images_std
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) /
                                (np.max(images_normalized) - minv)) * 255
    
    return images_normalized


def adjust_gamma(images, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :],
                                        dtype = np.uint8), table)

    return new_images


import random

class DataAugmenter:
    def __init__(self, contrast_factor, brightness_delta, gaussian_sigma, saturation_factor):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.gaussian_sigma = gaussian_sigma
        self.saturation_factor = saturation_factor
        self.gaussian_kernel = (11, 11)
        
        
    def __call__(self, image):
        """
        apply effects on images
        """
        if self.gaussian_sigma:
#             print('Adjusting Gaussian filter: ', self.gaussian_sigma)
            image = self.adjust_gaussian(image, self.gaussian_sigma)
#             print(image)
        
        if self.contrast_factor:
#             print('Adjusting Contrast: ', self.contrast_factor)
            image = self.adjust_contrast(image, self.contrast_factor)
            
        if self.brightness_delta:
#             print('Adjusting Brightness: ', self.brightness_delta)
            image = self.adjust_brightness(image, self.brightness_delta)
            
        if self.saturation_factor:
#             print('Adjusting Saturation: ', self.saturation_factor)
            image = self.adjust_saturation(image, self.saturation_factor)
            
        return image
    
    def _clip(self,image):
        """
        clip image data to 0~255 and convert to uint8 dtype
        """
        return np.clip(image, 0, 255).astype(np.uint8)
        
    def adjust_brightness(self, image, brightness):
        '''
        Randomly change the brightness of the input image.
        Protected against overflow.
        '''
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # To protect against overflow: Calculate a mask for all pixels
        # where adjustment of the brightness would exceed the maximum
        # brightness value and set the value to the maximum at those pixels.
        mask = hsv[:,:,2] * brightness > 255
        v_channel = np.where(mask, 255, hsv[:,:,2] * brightness)
        hsv[:,:,2] = v_channel
        
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#         image = cv2.cvtColor(hsv, cv2.COLOR_RGB2BGR)
        return image
    
    def adjust_contrast(self, image, factor):
#         print(image)
        mean = image.mean(axis=0).mean(axis=0)
        return self._clip((image - mean) * factor + mean)
    
    
    def adjust_saturation(self, image, factor):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image 
    
    def adjust_gaussian(self, img, sigma=3):
        img = cv2.GaussianBlur(img, self.gaussian_kernel, sigma)
        return img

def _random(val_range):
    return np.random.uniform(val_range[0], val_range[1]) 

    
def AugmentAgent(
        contrast_range=(0.7, 1.2),
        brightness_range=(.5, 1.5),
        gaussian_sigma_range=(1, 3),
        saturation_range=(0.95, 1.05)):
        
        def _generate():
            while True:
                yield DataAugmenter(
                    contrast_factor=_random(contrast_range),
                    brightness_delta=_random(brightness_range),
                    gaussian_sigma=_random(gaussian_sigma_range),
                    saturation_factor=_random(saturation_range)
                    )
            
        return _generate()
