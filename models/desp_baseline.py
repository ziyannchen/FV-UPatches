import cv2
import numpy as np


class RootSIFT:
    def __init__(self, eps=1e-7):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.eps = eps
    
    def detect(self, image):
        return self.sift.detect(image)
    
    def compute(self, image, kp):
        kps, descs = self.sift.compute(image, kp)
        if len(kps) == 0:
            return ([], None)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + self.eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return kps, descs


class DespBaseline:
    # Local matching baseline methods
    def __init__(self, atype='RootSIFT'):
        if atype is 'RootSIFT':
            self.baseline = RootSIFT()
            self.dist_thresh = 1.2
        elif atype is 'SURF':
            self.baseline = cv2.xfeatures2d.SURF_create()
            self.dist_thresh = 800
        elif atype is 'FAST':
            self.baseline = cv2.FastFeatureDetector_create()
            self.dist_thresh = 1.2
    
    def detect(self, image):
        kps = self.baseline.detect(image)
        return kps
    
    def compute(self, image, kp):
        kps, des = self.baseline.compute(image, kp)
        if len(kps) == 0:
            return ([], None)
        
        return kps, des