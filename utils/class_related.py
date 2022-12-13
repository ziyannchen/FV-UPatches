from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Matcher:
    def __init__(self, dist_thresh):
        self.dist_thresh = dist_thresh
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def fine_filter(self, goodMatch):
        # fine filter, sorting pairs according to their distance, filter out duplicate descriptorID with larger distance
        goodMatch = sorted(goodMatch, key = lambda x:x.distance)
        tmp = {}
        p1 = []; p2 = []
        gd = []
        for kpp in goodMatch:
            if str(kpp.queryIdx)+'_1' not in tmp and str(kpp.trainIdx)+'_2' not in tmp:
                p1.append(kpp.queryIdx)
                p2.append(kpp.trainIdx)
                tmp[str(kpp.queryIdx) + '_1'] = 1
                tmp[str(kpp.trainIdx) + '_2'] = 1
                gd.append(kpp)
            
        return gd, p1, p2
        
    def BFmatch(self, des1, des2):
        matches = self.bf.match(des1, des2)
        goodMatch = []
        # first coarse filter
        for m in matches:
            if abs(m.distance) < self.dist_thresh:
                goodMatch.append(m)
         
        gd, p1, p2 = self.fine_filter(goodMatch)
        return gd, p1, p2
    
    def KNNmatch(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k = 2)
        goodMatch = []
        # first coarse filter
        if len(matches) != 0 and len(matches[0])==2:
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    goodMatch.append(m)
                    
        return self.fine_filter(goodMatch)
    

def homographyMatch(p1, p2, kp1, kp2, goodMatch):
    match_num = 0
    good = []
    if len(goodMatch) >= 4: 
        src_pts = np.float32([kp1[p].pt for p in p1]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[p].pt for p in p2]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        for s, i in enumerate(mask):
            if i.squeeze()==1:
                good.append(goodMatch[s])

        match_num = mask.squeeze().sum()
        
    return match_num, good
        

def results_save(ls, nums):
    fpr, tpr, threshold = roc_curve(ls, nums)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)
    return fpr, tpr, eer, thresh, roc_auc


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, max(fper)], [0, max(tper)], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.ylim(0.95, 1.)
    plt.legend()
    plt.savefig('ROC.png')
    plt.show()